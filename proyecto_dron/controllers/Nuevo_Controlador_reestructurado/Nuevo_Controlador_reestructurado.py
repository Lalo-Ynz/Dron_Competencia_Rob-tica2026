from controller import Robot
import math
import numpy as np
import cv2

# Procesador de imagenes para la camara.
def image_to_mat(img_camara):
    img_data = img_camara.getImage()
    width = img_camara.getWidth()
    height = img_camara.getHeight()
    image = np.frombuffer(img_data, dtype=np.uint8).reshape((height, width, 4))
    return image[:, :, :3].copy()

# Auxiliar para mantener los numeros dentro del rango
def sign(x):
    return (x > 0) - (x < 0)
def clamp(value, low, high):
    return max(low, min(value, high))

# Deteccion de aros
def detectar_aros_cuadrados(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    edges=cv2.Canny(blur,70,150)
    contours,_=cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    candidatos=[]

    for cnt in contours:

        area=cv2.contourArea(cnt)
        if area<800: continue

        peri=cv2.arcLength(cnt,True)
        approx=cv2.approxPolyDP(cnt,0.03*peri,True)

        if len(approx)!=4: continue
        if not cv2.isContourConvex(approx): continue

        x,y,w,h=cv2.boundingRect(approx)

        ratio=w/(h+1e-6)
        if ratio<0.65 or ratio>1.35: continue

        area_box=w*h
        relleno=area/(area_box+1e-6)

        if relleno<0.45 or relleno>0.95: continue

        cx=x+w//2
        cy=y+h//2

        candidatos.append({"contorno":approx,"centro":(cx,cy),"area":area,"dist_centro":abs(cx-frame.shape[1]//2)+abs(cy-frame.shape[0]//2)})

    if len(candidatos)==0:
        return None

    candidatos=sorted(candidatos,key=lambda a:a["dist_centro"])

    return candidatos[0]

# Moverse en frente del aro
def control_visual_aro(aro, width, height):
    cx, cy = aro["centro"]
    area = aro["area"]  

    # Errores en píxeles relativo al centro de la imagen
    err_x = (cx - width // 2)
    err_y = (cy - height // 2)

    # Control de distancia: compara area del cuadrado detectado vs area de referencia
    area_ref = 3500  
    err_area = area_ref - area

    # Normalizar errores para que sean manejables 
    norm_err_x = err_x / float(width // 2)
    norm_err_y = err_y / float(height // 2)
    norm_err_area = err_area / area_ref

    # Ganancias a ajustar por prueba
    Kx = 1.0
    Ky = 1.0
    Ka = 1.0

    # Roll controla izquierda/derecha, Pitch es adelante/atrás
    roll_disturbance = Kx * clamp(norm_err_x, -0.3, 0.3)
    pitch_disturbance = Ky * clamp(-norm_err_y, -0.3, 0.3)  
    avance = Ka * clamp(norm_err_area, -1.0, 1.0)  
    return roll_disturbance, pitch_disturbance, avance

# --------------------- Máquina de estados para misiones ----------------------

MIS_INICIO = 0
MIS_POS_ARO = 1
MIS_GIRAR = 2
MIS_ARO = 3
MIS_ATRAVESAR_ARO = 4
MIS_POS_ARUCO = 5
MIS_ARUCO = 6  
MIS_PIZARRON = 7
MIS_PLATAFORMA = 8
MIS_ATERRIZAJE = 9

estado_mision = MIS_INICIO
estado_anterior = None 

# -------------------- Inicio de Programa -------------------------------------

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Sensores necesarios
camera = robot.getDevice("camera")
camera.enable(timestep)
imu = robot.getDevice("inertial unit")
imu.enable(timestep)
gyroscope = robot.getDevice("gyro")
gyroscope.enable(timestep)
compass = robot.getDevice("compass")
compass.enable(timestep)
gps = robot.getDevice("gps")
gps.enable(timestep)

# Helices y Actuadores
m1 = robot.getDevice('front left propeller')
m2 = robot.getDevice('front right propeller')
m3 = robot.getDevice('rear left propeller')
m4 = robot.getDevice('rear right propeller')

motores = [m1, m2, m3, m4]
for m in motores:
    m.setPosition(float('inf'))
    m.setVelocity(0.0)

# Motores de cámara
mcr = robot.getDevice('camera roll')
mcp = robot.getDevice('camera pitch')

display = robot.getDevice("display")

# Constantes
ALTURA_MANTENIDA = 2.0
VELOCIDAD_BASE = 100.0
TOLERANCIA = 0.03

k_vertical_thrust = 68.5    
k_roll_p = 50.0             
k_pitch_p = 30.0           
k_vertical_p = 5.0   
k_vertical_offset = 0.6
k_xy = 1.0
d_xy = 2.0  #Freno

TARGET_X1, TARGET_Y1, TARGET_Z1= -0.60, -3.32, 0.15
tiempo_estabilizacion = 3.0
inicio_timer= None

# Inicialización de velocidades (evita locals()) 
x_prev = 0.0
y_prev = 0.0
z_prev = 0.0

roll_disturbance = 0.0
pitch_disturbance = 0.0
yaw_input = 0.0
altura_objetivo = 1.65

#-------------------------------INICIO DEL WHILE -------------------------------------------

while robot.step(timestep) != -1:

    # Sensores 
    pos = gps.getValues()
    x, y, altitude = pos
    ang = imu.getRollPitchYaw()
    roll, pitch, yaw = ang
    gyro = gyroscope.getValues()
    r_vel, p_vel, y_vel = gyro

        # PROTECCIÓN GLOBAL ANTI-COMPLEX )
    if not math.isfinite(yaw):
        yaw = 0.0
    if not math.isfinite(roll):
        roll = 0.0
    if not math.isfinite(pitch):
        pitch = 0.0

    dt = timestep / 1000.0

    # Vision de la camara
    frame = image_to_mat(camera)
    frame_debug = frame.copy()
    aro_detectado = detectar_aros_cuadrados(frame)

    # Dibujar TODOS los contornos detectados
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 200:
            cv2.drawContours(frame_debug, [approx], -1, (0, 255, 0), 2)

            M = cv2.moments(approx)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(frame_debug, (cx, cy), 4, (0, 0, 255), -1)

    image_ref = display.imageNew(
        frame_debug.tobytes(),
        display.RGB,
        camera.getWidth(),
        camera.getHeight()
    )
    display.imagePaste(image_ref, 0, 0, False)

    # Velocidad vertical 
    vz = (altitude - z_prev) / dt
    
    # Velocidad horizontal 
    vx = (x - x_prev) / dt
    vy = (y - y_prev) / dt

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    vx_body = cos_yaw * vx + sin_yaw * vy
    vy_body = -sin_yaw * vx + cos_yaw * vy

    x_prev = x
    y_prev = y
    z_prev = altitude

    # Estabilización de cámara 
    mcr.setPosition(-0.115 * r_vel)
    mcp.setPosition(-0.1 * p_vel)

    # Misiones ordenadas por bloques
    if estado_mision == MIS_INICIO:

        if estado_anterior != MIS_INICIO:
            x_set = x
            y_set = y
            yaw_set = yaw
            inicio_timer = None
            estado_anterior = MIS_INICIO
        altura_objetivo = 1.65

        # ERRORES 
        err_x = x_set - x
        err_y = y_set - y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        err_x_body = cos_yaw * err_x + sin_yaw * err_y
        err_y_body = -sin_yaw * err_x + cos_yaw * err_y

        # CONTROL HORIZONTAL 
        vel_x_deseada = clamp(err_x_body * 2.0, -1.0, 1.0)
        vel_y_deseada = clamp(err_y_body * 2.0, -1.0, 1.0)

        pitch_disturbance = clamp((vx_body - vel_x_deseada) * 3.0, -2.0, 2.0)
        roll_disturbance  = clamp(-(vy_body - vel_y_deseada) * 3.0, -2.0, 2.0)

        if altitude < 0.2:
            factor = clamp((altitude - 0.3) / 0.5, 0.0, 1.0)
            pitch_disturbance *= factor
            roll_disturbance  *= factor

        # YAW 
        yaw_error = math.atan2(math.sin(yaw_set - yaw), math.cos(yaw_set - yaw))
        yaw_input = clamp(yaw_error * 2.0, -1.0, 1.0) - y_vel

        # TRANSICIÓN 
        if abs(altitude - altura_objetivo) < 0.05:
            if inicio_timer is None:
                inicio_timer = robot.getTime()

            if robot.getTime() - inicio_timer >= tiempo_estabilizacion:
                print("Altura alcanzada")
                estado_mision = MIS_POS_ARO
        else:
            inicio_timer = None


    elif estado_mision == MIS_POS_ARO:

        # DETECCIÓN DE ENTRADA 
        if estado_anterior != MIS_POS_ARO:
            yaw_fijo = yaw
            timer_frenado = None
            estado_anterior = MIS_POS_ARO

        altura_objetivo = 1.65

        # YAW 
        yaw_error = math.atan2(math.sin(yaw_fijo - yaw), math.cos(yaw_fijo - yaw))
        yaw_input = clamp(yaw_error * 2.0, -1.0, 1.0) - y_vel

        # POSICIÓN 
        target_x, target_y = -0.798, -3.33

        err_x_global = target_x - x
        err_y_global = target_y - y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
        err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global

        distancia = math.sqrt(err_x_global**2 + err_y_global**2)

        gain_p = 1.0 if distancia > 0.15 else 0.4

        vel_x_deseada = clamp(err_x_body * gain_p, -0.6, 0.6)
        vel_y_deseada = clamp(err_y_body * gain_p, -0.6, 0.6)

        pitch_disturbance = clamp((vx_body - vel_x_deseada) * 3.0, -2.0, 2.0)
        roll_disturbance  = clamp(-(vy_body - vel_y_deseada) * 3.0, -2.0, 2.0)

        # TRANSICIÓN MEJORADA
        cond_pos = distancia < 0.12
        cond_vel = abs(vx_body) < 0.05 and abs(vy_body) < 0.05

        if cond_pos and cond_vel:
            if timer_frenado is None:
                timer_frenado = robot.getTime()

            if robot.getTime() - timer_frenado >= 1.5:
                timer_frenado = None
                estado_mision = MIS_GIRAR
        else:
            timer_frenado = None


    elif estado_mision == MIS_GIRAR:
        # DETECCIÓN DE ENTRADA
        if estado_anterior != MIS_GIRAR:
            yaw_inicio = yaw
            yaw_objetivo = yaw_inicio + 1.57  # giro relativo de 90°
            timer_estable_giro = None
            mensaje_giro_impreso = False
            estado_anterior = MIS_GIRAR

        altura_objetivo = 1.65

        # YAW
        yaw_error = math.atan2(math.sin(yaw_objetivo - yaw), math.cos(yaw_objetivo - yaw))
        yaw_input = clamp((1.2 * yaw_error) - (0.8 * y_vel), -1.0, 1.0)

        # POSICIÓN (ANCLADO)
        target_x, target_y = -0.691, -3.33

        err_x_global = target_x - x
        err_y_global = target_y - y

        # YAW SUAVIZADO
        yaw_error = math.atan2(math.sin(yaw_objetivo - yaw), math.cos(yaw_objetivo - yaw))

        # desaceleración progresiva cerca del objetivo
        factor_suavizado = clamp(abs(yaw_error) / 1.57, 0.25, 1.0)

        yaw_input = ((1.05 * yaw_error * factor_suavizado) - (1.6* y_vel))

        yaw_input = clamp(yaw_input, -0.9, 0.9)

        err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
        err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global

        vel_x_deseada = clamp(err_x_body * 1.2, -0.25, 0.25)
        vel_y_deseada = clamp(err_y_body * 1.2, -0.25, 0.25)

        pitch_disturbance = clamp((vx_body - vel_x_deseada) * 1.8, -0.8, 0.8)
        roll_disturbance  = clamp(-(vy_body - vel_y_deseada) * 1.8, -0.8, 0.8)

        # ESTABILIZACIÓN
        cond_altura = abs(altitude - altura_objetivo) < 0.05
        cond_yaw = abs(yaw_error) < 0.03 and abs(y_vel) < 0.05

        if cond_altura and cond_yaw:
            if timer_estable_giro is None:
                timer_estable_giro = robot.getTime()

            if robot.getTime() - timer_estable_giro >= 1.5:
                if not mensaje_giro_impreso:
                    print("Giro estabilizado")
                    mensaje_giro_impreso = True
                    

                vertical_input_post_giro = vertical_input
                roll_input_post_giro = roll_input
                pitch_input_post_giro = pitch_input
                yaw_post_giro = yaw  
                estado_mision = MIS_ARO

    elif estado_mision==MIS_ARO:

        if estado_anterior!=MIS_ARO:
            yaw_fijo=yaw
            timer_foto=None
            estado_anterior=MIS_ARO
            detecciones_consecutivas=0
            x_hold=x
            y_hold=y

            try:
                cv2.namedWindow("Foto_Aros",cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Foto_Aros",900,700)
            except:
                pass

        altura_objetivo=1.65

        yaw_error=math.atan2(math.sin(yaw_fijo-yaw),math.cos(yaw_fijo-yaw))
        yaw_input=clamp((2.2*yaw_error)-(0.6*y_vel),-1.2,1.2)

        # MANTENERSE QUIETO
        err_x_hold=x_hold-x
        err_y_hold=y_hold-y

        cos_yaw=math.cos(yaw)
        sin_yaw=math.sin(yaw)

        err_x_body=cos_yaw*err_x_hold+sin_yaw*err_y_hold
        err_y_body=-sin_yaw*err_x_hold+cos_yaw*err_y_hold

        vel_x_hold=clamp(err_x_body*2.0,-0.05,0.05)
        vel_y_hold=clamp(err_y_body*2.0,-0.05,0.05)

        pitch_disturbance=clamp((vx_body-vel_x_hold)*6.0,-0.3,0.3)
        roll_disturbance=clamp(-(vy_body-vel_y_hold)*6.0,-0.3,0.3)

        # VISIÓN
        frame_debug=frame.copy()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)

        edges=cv2.Canny(blur,80,160)

        contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        mejor=None
        mejor_score=999999

        for cnt in contours:

            area=cv2.contourArea(cnt)

            if area<400:
                continue

            perimeter=cv2.arcLength(cnt,True)

            approx=cv2.approxPolyDP(cnt,0.015*perimeter,True)

            if len(approx)!=4:
                continue

            x1,y1,w1,h1=cv2.boundingRect(approx)

            if w1<80 or h1<80:
                continue

            if w1>frame.shape[1]*0.7 or h1>frame.shape[0]*0.7:
                continue

            ratio=w1/float(h1)

            if ratio<0.75 or ratio>1.25:
                continue

            cx=x1+w1//2
            cy=y1+h1//2

            dist_centro=abs(cx-frame.shape[1]//2)

            score=dist_centro-(area*0.01)

            if score<mejor_score:
                mejor_score=score
                mejor=(approx,cx,cy,x1,y1,w1,h1)

        if mejor is not None:

            detecciones_consecutivas+=1

            approx,cx,cy,x1,y1,w1,h1=mejor

            cv2.drawContours(frame_debug,[approx],-1,(0,255,0),3)

            cv2.circle(frame_debug,(cx,cy),6,(0,0,255),-1)

            cv2.putText(frame_debug,"ARO DETECTADO",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        else:

            detecciones_consecutivas=0

            cv2.putText(frame_debug,"SIN DETECCION",(20,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        edges_bgr=cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)

        debug_total=np.hstack((frame_debug,edges_bgr))

        try:
            cv2.imshow("Foto_Aros",debug_total)

            if cv2.getWindowProperty("Foto_Aros",cv2.WND_PROP_VISIBLE)>=1:
                cv2.waitKey(1)

        except:
            pass

        if detecciones_consecutivas>=15:

            if timer_foto is None:
                timer_foto=robot.getTime()

            elif robot.getTime()-timer_foto>1.0:

                try:
                    cv2.destroyWindow("Foto_Aros")
                except:
                    pass

                print("Aro Confirmado")

                estado_mision=MIS_ATRAVESAR_ARO

        else:
            timer_foto=None

    elif estado_mision == MIS_ATRAVESAR_ARO:
            
    # DETECCIÓN DE ENTRADA
            if estado_anterior != MIS_ATRAVESAR_ARO:
                yaw_fijo = yaw
                timer_frenado = None
                estado_anterior = MIS_ATRAVESAR_ARO

            altura_objetivo = 1.65

            # YAW
            yaw_error = math.atan2(math.sin(yaw_fijo - yaw), math.cos(yaw_fijo - yaw))
            yaw_input = clamp((2.2 * yaw_error) - (0.5 * y_vel), -1.8, 1.8)

            # POSICIÓN
            target_x, target_y = -0.798, 3.86

            err_x_global = target_x - x
            err_y_global = target_y - y

            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)

            err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
            err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global

            distancia = math.sqrt(err_x_global**2 + err_y_global**2)

            gain_p = 1.0 if distancia > 0.15 else 0.4

            # PRIORIDAD AL AVANCE
            vel_x_deseada = clamp(err_x_body * 1.4, -0.45, 0.45)
            vel_y_deseada = clamp(err_y_body * 0.8, -0.18, 0.18)

            if abs(vy_body) < 0.015:
                vy_body = 0.0

            # CONTROL AMORTIGUADO
            pitch_disturbance = clamp((vx_body - vel_x_deseada) * 2.2, -1.2, 1.2)
            roll_disturbance = clamp(-(vy_body - vel_y_deseada) * 1.5, -0.7, 0.7)

            # TRANSICIÓN MEJORADA 
            cond_pos = distancia < 0.12
            cond_vel = abs(vx_body) < 0.05 and abs(vy_body) < 0.05

            if cond_pos and cond_vel:
                if timer_frenado is None:
                    timer_frenado = robot.getTime()

                if robot.getTime() - timer_frenado >= 1.5:
                    timer_frenado = None
                    print("Prueba 1 Superada")
                    estado_mision = MIS_POS_ARUCO
            else:
                timer_frenado = None

    elif estado_mision == MIS_POS_ARUCO:
            
        if estado_anterior != MIS_POS_ARUCO:
            yaw_inicio = yaw
            yaw_objetivo = yaw_inicio - 1.57  # giro relativo de 90°
            timer_estable_giro = None
            estado_anterior = MIS_POS_ARUCO

        altura_objetivo = 1.65

        # YAW SUAVIZADO
        yaw_error = math.atan2(math.sin(yaw_objetivo - yaw), math.cos(yaw_objetivo - yaw))
        factor_suavizado = clamp(abs(yaw_error) / 1.57, 0.25, 1.0)

        yaw_input = ((1.05 * yaw_error * factor_suavizado) - (1.6 * y_vel))
        yaw_input = clamp(yaw_input, -0.75, 0.75)

        # POSICIÓN (ANCLADO)
        target_x, target_y = -0.78, 3.86

        err_x_global = target_x - x
        err_y_global = target_y - y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
        err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global

        vel_x_deseada = clamp(err_x_body * 1.2, -0.25, 0.25)
        vel_y_deseada = clamp(err_y_body * 1.2, -0.25, 0.25)

        pitch_disturbance = clamp((vx_body - vel_x_deseada) * 1.8, -0.8, 0.8)
        roll_disturbance  = clamp(-(vy_body - vel_y_deseada) * 1.8, -0.8, 0.8)

        # ESTABILIZACIÓN
        cond_altura = abs(altitude - altura_objetivo) < 0.05
        cond_yaw = abs(yaw_error) < 0.03 and abs(y_vel) < 0.05

        if cond_altura and cond_yaw:
            if timer_estable_giro is None:
                timer_estable_giro = robot.getTime()

            if robot.getTime() - timer_estable_giro >= 1.5:    
                vertical_input_post_giro = vertical_input
                roll_input_post_giro = roll_input
                pitch_input_post_giro = pitch_input
                yaw_post_giro = yaw  
                estado_mision = MIS_ARUCO

    elif estado_mision==MIS_ARUCO:
        if estado_anterior!=MIS_ARUCO:
            yaw_fijo=yaw
            mensaje_aruco_alcanzado=False
            timer_aruco=None

            aruco_dict=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
            aruco_params=cv2.aruco.DetectorParameters()
            aruco_params.adaptiveThreshWinSizeMin=3
            aruco_params.adaptiveThreshWinSizeMax=23
            aruco_params.minMarkerPerimeterRate=0.02
            aruco_params.maxMarkerPerimeterRate=4.0
            detector=cv2.aruco.ArucoDetector(aruco_dict,aruco_params)
            estado_anterior=MIS_ARUCO

        altura_objetivo=1.65
        yaw_error=math.atan2(math.sin(yaw_fijo-yaw),math.cos(yaw_fijo-yaw))
        yaw_input=clamp(yaw_error*2.0-y_vel,-1.2,1.2)

        h,w=frame.shape[:2]
        frame_zoom=frame.copy()
        zoom=2.2

        nw=int(w/zoom)
        nh=int(h/zoom)
        
        x0=(w-nw)//2
        y0=(h-nh)//2

        frame_zoom=frame_zoom[y0:y0+nh,x0:x0+nw]
        frame_zoom=cv2.resize(frame_zoom,(w,h))
        corners,ids,_=detector.detectMarkers(frame_zoom)

        roll_disturbance=0.0
        pitch_disturbance=0.0
        detectado=False

        if ids is not None:
            for i,mid in enumerate(ids.flatten()):
                if mid!=100: continue
                detectado=True
                pts=corners[i][0]

                cx=int(np.mean(pts[:,0]))
                cy=int(np.mean(pts[:,1]))
                
                ancho=max(np.linalg.norm(pts[0]-pts[1]),np.linalg.norm(pts[2]-pts[3]))
                distancia=(0.1*420)/(ancho+1)

                err_x=(cx-w*0.5)/(w*0.5)
                err_y=(cy-h*0.5)/(h*0.5)

                objetivo=0.08
                err_dist=distancia-objetivo
                vel_frente=clamp(err_dist*3.0,-0.60,0.60)

                if distancia<0.35: vel_frente*=0.70
                if distancia<0.22: vel_frente*=0.45
                if distancia<0.15: vel_frente*=0.20

                vel_obj=vel_frente
                roll_disturbance=clamp(-err_x*2.2,-0.55,0.55)
                pitch_disturbance=clamp((vx_body-vel_obj)*3.5+err_y*0.8,-1.3,1.3)

                if distancia<0.12:
                    pitch_disturbance-=vx_body*10.0
                    roll_disturbance-=vy_body*10.0
                break

        if not detectado:
            target_x=4.5
            target_y=3.7

            err_x_global=target_x-x
            err_y_global=target_y-y

            cos_yaw=math.cos(yaw)
            sin_yaw=math.sin(yaw)

            err_x_body=cos_yaw*err_x_global+sin_yaw*err_y_global
            err_y_body=-sin_yaw*err_x_global+cos_yaw*err_y_global

            vel_x_deseada=clamp(err_x_body*1.4,-0.55,0.55)
            vel_y_deseada=clamp(err_y_body*1.4,-0.55,0.55)

            pitch_disturbance=clamp((vx_body-vel_x_deseada)*3.5,-1.3,1.3)
            roll_disturbance=clamp(-(vy_body-vel_y_deseada)*3.5,-1.3,1.3)

        cond_movimiento_cero = abs(vx_body) < 0.05 and abs(vy_body) < 0.05
        
        if cond_movimiento_cero:
            if timer_aruco is None:
                timer_aruco = robot.getTime()
            elif robot.getTime() - timer_aruco >= 2.0:
                roll_disturbance = 0.0
                pitch_disturbance = 0.0
                if not mensaje_aruco_alcanzado:
                    print("ARUCO alcanzado")
                    mensaje_aruco_alcanzado = True
                    estado_mision = MIS_PIZARRON
        else:
            timer_aruco = None

    elif estado_mision == MIS_PIZARRON:
        if estado_anterior != MIS_PIZARRON:
            yaw_fijo = yaw
            timer_pizarron = None
            estado_anterior = MIS_PIZARRON

        altura_objetivo = 1.65

        yaw_error = math.atan2(math.sin(yaw_fijo - yaw), math.cos(yaw_fijo - yaw))
        yaw_input = clamp(yaw_error * 2.0 - y_vel, -1.2, 1.2)

        target_x, target_y = 4.55, 1.85

        err_x_global = target_x - x
        err_y_global = target_y - y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
        err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global

        distancia = math.sqrt(err_x_global**2 + err_y_global**2)

        vel_x_deseada = clamp(err_x_body * 1.5, -0.45, 0.45)
        vel_y_deseada = clamp(err_y_body * 0.8, -0.18, 0.18)

        if abs(vy_body) < 0.015:
            vy_body = 0.0

        pitch_disturbance = clamp((vx_body - vel_x_deseada) * 2.3, -1.0, 1.0)
        roll_disturbance = clamp(-(vy_body - vel_y_deseada) * 1.5, -0.65, 0.65)
        
        cond_pos = distancia < 0.10
        cond_vel = abs(vx_body) < 0.05 and abs(vy_body) < 0.05

        if cond_pos and cond_vel:
            if timer_pizarron is None:
                timer_pizarron = robot.getTime()
            elif robot.getTime() - timer_pizarron >= 1.5:
                print("Prueba 2 Superada")
                estado_mision = MIS_PLATAFORMA
        else:
            timer_pizarron = None

    elif estado_mision == MIS_PLATAFORMA:
        if estado_anterior != MIS_PLATAFORMA:
            yaw_fijo = yaw
            timer_plataforma = None
            mensaje_plataforma = False
            estado_anterior = MIS_PLATAFORMA

        altura_objetivo = 1.65
        
        yaw_error = math.atan2(math.sin(yaw_fijo - yaw), math.cos(yaw_fijo - yaw))
        yaw_input = clamp(yaw_error * 2.0 - y_vel, -1.2, 1.2)
        target_x, target_y = 3.82, -3.24

        err_x_global = target_x - x
        err_y_global = target_y - y

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        err_x_body = cos_yaw * err_x_global + sin_yaw * err_y_global
        err_y_body = -sin_yaw * err_x_global + cos_yaw * err_y_global
        
        distancia = math.sqrt(err_x_global**2 + err_y_global**2)
        gain_p = 1.4 if distancia > 0.20 else 0.5

        vel_x_deseada = clamp(err_x_body * gain_p, -0.55, 0.55)
        vel_y_deseada = clamp(err_y_body * gain_p, -0.55, 0.55)

        pitch_disturbance = clamp((vx_body - vel_x_deseada) * 3.5, -1.3, 1.3)
        roll_disturbance = clamp(-(vy_body - vel_y_deseada) * 3.5, -1.3, 1.3)

        cond_pos = distancia < 0.10
        cond_vel = abs(vx_body) < 0.05 and abs(vy_body) < 0.05

        if cond_pos and cond_vel:
            if timer_plataforma is None:
                timer_plataforma = robot.getTime()
            elif robot.getTime() - timer_plataforma >= 3.0:
                if not mensaje_plataforma:
                    print("Posicionado")
                    mensaje_plataforma = True
                    estado_mision = MIS_ATERRIZAJE
        else:
            timer_plataforma = None
        
    elif estado_mision == MIS_ATERRIZAJE:

        if estado_anterior != MIS_ATERRIZAJE:
            timer_estabilizacion = None
            mensaje_apagado = False
            estado_anterior = MIS_ATERRIZAJE

        m1.setVelocity(0.0)
        m2.setVelocity(0.0)
        m3.setVelocity(0.0)
        m4.setVelocity(0.0)
        
 # --- CONTROL GLOBAL  ---

    # ALTURA
    clamped_difference_altitude = clamp(altura_objetivo - altitude + k_vertical_offset,-1.0, 1.0)

    vertical_input = k_vertical_p * (clamped_difference_altitude ** 4) - (5.0 * vz)

    # CONTROL FINAL
    roll_input  = k_roll_p * clamp(roll, -1.0, 1.0) + r_vel + roll_disturbance
    pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + p_vel + pitch_disturbance

        # PROTECCIÓN FUERTE ANTI-COMPLEX
    if 'roll_disturbance' not in locals() or not math.isfinite(roll_disturbance):
        roll_disturbance = 0.0
    if 'pitch_disturbance' not in locals() or not math.isfinite(pitch_disturbance):
        pitch_disturbance = 0.0
    if 'yaw_input' not in locals() or not math.isfinite(yaw_input):
        yaw_input = 0.0

    # Forzar que vertical_input también sea real
    if 'vertical_input' not in locals() or not math.isfinite(vertical_input):
        vertical_input = 0.0

    if estado_mision != MIS_ATERRIZAJE:
        thrust_actual = k_vertical_thrust

    # MOTORES
        if estado_mision == MIS_ATERRIZAJE:
            m1.setVelocity(0.0)
            m2.setVelocity(0.0)
            m3.setVelocity(0.0)
            m4.setVelocity(0.0)
            roll_input = 0.0
            pitch_input = 0.0
        else: 
            m1.setVelocity(thrust_actual + vertical_input - roll_input + pitch_input - yaw_input)
            m2.setVelocity(-(thrust_actual + vertical_input + roll_input + pitch_input + yaw_input))
            m3.setVelocity(-(thrust_actual + vertical_input - roll_input - pitch_input + yaw_input))
            m4.setVelocity(thrust_actual + vertical_input + roll_input - pitch_input - yaw_input)

    #-----------------------FIN DEL WHILE -------------------------------------
