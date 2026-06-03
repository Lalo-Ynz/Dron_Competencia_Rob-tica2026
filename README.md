Contexto de la Competencia
Este repositorio contiene el sistema de control de vuelo autónomo para un microvehículo aéreo desarrollado en el simulador Webots. El proyecto se alinea con el reglamento de la categoría de Drones Autónomos del Torneo Mexicano de Robótica 2026 y las normativas internacionales del IMAV orientadas a misiones de búsqueda y rescate en interiores. El software opera de manera local a bordo de la aeronave, ejecutando el despegue, la navegación tridimensional, el procesamiento visual y el aterrizaje sin ninguna intervención humana o mando a distancia.

2. Misiones del Entorno
El algoritmo resuelve en secuencia las siguientes pruebas del reglamento:

Misión uno, ingreso a la habitación: Despegue automatizado y cruce del túnel de aros cuadrados de cincuenta centímetros por cincuenta centímetros mediante centrado óptico, obteniendo el máximo puntaje de la prueba.

Misión dos, trazo en la pizarra: Navegación guiada por coordenadas absolutas para posicionar al dron en un vuelo estacionario paralelo y controlado frente a la estructura de la pizarra de dibujo.

Misión tres, localización de objetivos: Activación de algoritmos de visión para identificar el marcador fiducial ArUco con el identificador numérico cien, estimando la distancia física para regular la velocidad de aproximación frontal.

Misión cuatro, aterrizaje en plataforma: Desplazamiento hacia el cuadrante de descenso final empleando un perfil de frenado proporcional adaptativo para mitigar la inercia y aplicar un corte total de energía sobre el centro de la base.

3. Arquitectura del Software
El controlador se ejecuta de forma cíclica dentro del bucle del simulador y está integrado por los siguientes módulos lógicos:

Adquisición sensorial y telemetría: Lectura de datos del GPS, la unidad de medición inercial y el giroscopio, transformando las velocidades globales al sistema de referencia local del cuerpo del dron mediante matrices de rotación trigonométricas.

Filtro de seguridad inercial: Mecanismo que intercepta valores numéricos infinitos o indefinidos provocados por colisiones o errores físicos de simulación, forzándolos a cero para mantener la estabilidad de los lazos de control.

Procesamiento y detección visual: Conversión del flujo de la cámara a matrices compatibles con OpenCV. Aplica escala de grises, desenfoque gaussiano y binarización de Canny para filtrar contornos por área, número de vértices, convexidad y solidez, aislando el objetivo.

Leyes de navegación por bloques: Rutinas secuenciales de control proporcional acopladas al giroscopio que regulan el despegue, la aproximación por área visual, giros puros en el eje de guiñada, contadores de persistencia temporal y frenado adaptativo.

Mezcla de potencia: Lazo final que combina los comandos de empuje, alabeo, cabeceo y guiñada mediante una matriz de mezcla para configuraciones de multirrotores en cruz, asignando de forma directa las velocidades a los cuatro actuadores del dron.
