import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Empty
import numpy as np
import cv2 as cv
import math as mt
from controller import Robot
import time
from std_msgs.msg import Float32MultiArray, Empty 
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry



def clamp(value,low,high):
    return max(low, min(value, high))

def apagar_motores(motores): 
    for m in motores:
        m.setVelocity(0.0)

class MavicController(Node):
    def __init__(self):

        super().__init__('mavic_controller')
        self.cmd_vel_sub=self.create_subscription(Twist,'/cmd_vel',self.cmd_vel_callback,10)
        self.takeoff_sub=self.create_subscription(Empty,'/takeoff',self.takeoff_callback,10)
        self.land_sub=self.create_subscription(Empty,'/land',self.land_callback,10)
        self.odom_pub=self.create_publisher(Odometry,'/odom',10)
        self.tf_broadcaster=TransformBroadcaster(self)


        self.robot= Robot()
        self.timestep=int(self.robot.getBasicTimeStep())
        self.last_time = self.robot.getTime()
        self.m1=self.robot.getDevice('front left propeller')
        self.m2=self.robot.getDevice('front right propeller')
        self.m3=self.robot.getDevice('rear left propeller')
        self.m4=self.robot.getDevice('rear right propeller')

        self.motors=[self.m1,self.m2,self.m3,self.m4]

        for m in self.motors:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
    

        self.flying=False 
        self.target_altitude=1.0   
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0
        self.hover_x = None 
        self.hover_y= None
        self.locked_yaw= None

        #seguimiento 
        self.ruta=[]
        self.ruta_act=0
        self.autom=False 
     

        self.imu=self.robot.getDevice('inertial unit')
        self.imu.enable(self.timestep)
        self.gyro=self.robot.getDevice('gyro')
        self.gyro.enable(self.timestep)
        self.gps=self.robot.getDevice('gps')
        self.gps.enable(self.timestep)
        self.depth_camera=self.robot.getDevice('depth_camera')
        self.depth_camera.enable(self.timestep)

        self.depth_width=self.depth_camera.getWidth()
        self.depth_height=self.depth_camera.getHeight()

        self.depth_pub=self.create_publisher(Float32MultiArray,'/camera/depth/raw_array',10)
        self.vision_timer=self.create_timer(0.2,self.publish_depth_data)

        self.target_vx=0.0
        self.target_vy=0.0
        self.target_vz=0.0
        self.target_yaw=0.0
        self.target_altitude = 1.0
        self.last_x = 0.0
        self.last_y = 0.0
        self.last_z = 0.0


        

    
    def run(self):
    # Este bucle sustituye al spin normal de ROS
        while self.robot.step(self.timestep) != -1:
            rclpy.spin_once(self, timeout_sec=0.0)
            self.control()


    def cmd_vel_callback(self, msg):
        
        self.target_vx=msg.linear.x
        self.target_vy=msg.linear.y
        self.target_vz=msg.linear.z
        self.target_yaw=msg.angular.z


    def takeoff_callback(self, msg):
        self.flying = True
        self.target_altitude = 1.5  # Altura de vuelo fija
        
        self.autom = True


    def check_landing(self):
        _,_,z=self.gps.getValues()
        if z <= 0.15:
            self.get_logger().info("Aterrizaje completado")
            self.landing_timer.cancel()
            self.flying=False
            apagar_motores(self.motors)
        else:
            self.get_logger().info(f"Altitud actual: {z:.2f} m")

    def land_callback(self, msg):
       
        self.target_altitude = 0.15
        self.landing_timer=self.create_timer(0.5,self.check_landing)

        
        

    def publish_depth_data(self):
        depth_array=self.depth_camera.getRangeImage()
        if not depth_array:
            return
        
        depthnp=np.array(depth_array,dtype=np.float32)
        depthnp[np.isinf(depthnp)]=0.0

        msg=Float32MultiArray()
        msg.data=depthnp.tolist()
        self.depth_pub.publish(msg)

    def euler_quat(self, roll, pitch, yaw):

        cy=mt.cos(yaw * 0.5)
        sy=mt.sin(yaw * 0.5)
        cp=mt.cos(pitch * 0.5)
        sp=mt.sin(pitch * 0.5)
        cr=mt.cos(roll * 0.5)
        sr=mt.sin(roll * 0.5)

        q=[0]*4
        q[0]=sr*cp*cy - cr*sp*sy
        q[1]=cr*sp*cy + sr*cp*sy
        q[2]=cr*cp*sy - sr*sp*cy
        q[3]=cr*cp*cy + sr*sp*sy
        return q

    def control(self):
        
        #self.robot.step(int(self.timestep))
    
        current_time = self.robot.getTime()
        dt = current_time - self.last_time
    
        if dt <= 0.0:
            return 
        
        self.last_time = current_time

        roll, pitch,yaw = self.imu.getRollPitchYaw()
        rv, pv, zv = self.gyro.getValues()
        x, y, z = self.gps.getValues()
        depth_data=self.depth_camera.getRangeImage()
        #self.get_logger().info(f"Posición: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        #self.get_logger().info(f"Orientación: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
        vx = (x - self.last_x) / dt
        vy = (y - self.last_y) / dt
        vz = (z - self.last_z) / dt
        cos_yaw = mt.cos(yaw)
        sin_yaw = mt.sin(yaw)
        vx_body = vx * cos_yaw + vy * sin_yaw
        vy_body = -vx * sin_yaw + vy * cos_yaw

        t= TransformStamped()
        t.header.stamp=self.get_clock().now().to_msg()
        t.header.frame_id='odom'
        t.child_frame_id='base_link'
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        q=self.euler_quat(roll, pitch, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        self.tf_broadcaster.sendTransform(t)

        #Odometria 
        odom_msg=Odometry()
        odom_msg.header.stamp=self.get_clock().now().to_msg()
        odom_msg.header.frame_id='odom'
        odom_msg.child_frame_id='base_link'
        odom_msg.pose.pose.position.x = x
        odom_msg.pose.pose.position.y = y
        odom_msg.pose.pose.position.z = z
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]
        self.odom_pub.publish(odom_msg)
        

        self.last_x = x
        self.last_y = y
        self.last_z = z

        #GANANCIAS DE PID 
        k_vertical_thrust = 68.5
        k_vertical_offset = 0.6
        k_vertical_p = 6.0
        k_roll_p = 50.0
        k_pitch_p = 30.0

        
        if self.flying is True:

                    # 1. CONTROL DE CASCADA: Posición -> Velocidad
                    if self.target_vx == 0.0 and self.target_vy == 0.0 and self.target_yaw == 0.0:
                        # Modo Hover: Anclamos la posición actual
                        if self.hover_x is None or self.hover_y is None:
                            self.hover_x = x
                            self.hover_y = y

                        err_x_global = self.hover_x - x
                        err_y_global = self.hover_y - y
                        err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
                        err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global
                        
                        # Traducimos el error de posición a una velocidad interna para regresar al punto
                        vel_x_deseada = clamp(err_x_body * 2.0, -1.0, 1.0)
                        vel_y_deseada = clamp(err_y_body * 2.0, -1.0, 1.0)
                    else:
                        # Modo Movimiento: Borramos el ancla y usamos el Twist de ROS
                        self.hover_x = None
                        self.hover_y = None
                        vel_x_deseada = self.target_vx
                        vel_y_deseada = self.target_vy

                    # 2. CONTROL ÚNICO DE VELOCIDAD (Frena y corrige automáticamente)
                    dist_pitch = clamp((vx_body - vel_x_deseada) * 3.0, -2.0, 2.0)
                    
                    # ¡CRÍTICO! El signo negativo al inicio de dist_roll es obligatorio para que no se vuelque
                    dist_roll = clamp(-(vy_body - vel_y_deseada) * 3.0, -2.0, 2.0)

                    # 3. SUMA FINAL
                    roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + rv + dist_roll
                    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pv + dist_pitch
                    yaw_input = self.target_yaw - zv

                    if self.target_yaw==0.0:
                        if self.locked_yaw is None:
                            self.locked_yaw = yaw
                        yaw_error=self.locked_yaw - yaw
                        yaw_error=mt.atan2(mt.sin(yaw_error), mt.cos(yaw_error))  # Normaliza el error a [-pi, pi]
                        yaw_input = clamp(yaw_error * 2.0, -1.0, 1.0)- zv
                    else:
                        self.locked_yaw = None
                        yaw_input = self.target_yaw - zv

                    clamped_difference_altitude = clamp(self.target_altitude - z + k_vertical_offset, -1.0, 1.0)
                    vertical_input = k_vertical_p * (clamped_difference_altitude ** 3.0) - (5.0 * vz)

                    self.m1.setVelocity(k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input)
                    self.m2.setVelocity(-(k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input))
                    self.m3.setVelocity(-(k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input))
                    self.m4.setVelocity(k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input)
                    self.target_altitude += self.target_vz * dt

def main(args=None):
    rclpy.init(args=args)
    mavic_controller=MavicController()
    mavic_controller.run()
    #rclpy.spin(mavic_controller)
    mavic_controller.destroy_node()
    rclpy.shutdown()
    


    
if __name__=='__main__':
    main()