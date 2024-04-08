
## `TODO PPT INDRA:`
- ¿Cual es el costo asociado de contratar a un reemplazo?
- ¿Cual es el costo asociado a realizar evaluaciones de rendimiento cada 3 meses?
- Variables con relacion negativa/positiva a la prediccion
- BI con indicadores de probabilidad
- Graficos
    - Como abordamos los valores nulos
    - Analisis de datos (presencia de outliers)
        - Lo que verbalmente sea dificil de explicar pero que de forma visual sea mas explicito
            - El grafo de jefes-empleados para explicar la variable `degree` que aporta a la prediccion bajo la hipotesis de que manejar equipos grandes no es lo mismo que manejar equipos chicos.
- Reflexionar acerca de los puntos desafiantes en el reto, lecciones aprendidas
    - Que harian directamente o mejorarian con lo que han presentado?
        - Entender mejor el negocio
            - Entender mejor como funcionan los equipos en Indra para generar nuevas variables o drivers
- Analisis de performance del modelo a posterior
    - Que el modelo prediga que un candidato no se va a los 6 meses, pero se va (esto se puede medir a posterior)
    - Que el modelo prediga que un candidato se va a los 6 meses (no hay forma practica de saber si efectivamente el modelo acerto)
    - Matriz de confusion
        - El modelo minimiza falsos negativos
        - Sacrificamos por falsos positivos dado que a partir de la prediccion y el valor de probabilidad se puede hacer un analisis mas minucioso caso a caso
    - Marco teorico del modelo (XGBoost)
        - Bagging
        - Boosting
        - Bootstrap
        - Regularization
        - Missing value handling

        - Tiempo de entrenamiento
        - Tiempo de inferencia
    - Resultados de cross validation (mean, std) 
- Mapa mental del trabajo
    - Puntos fuertes
    - Aspectos de mejora
        - En funcion a los grafos que creamos podriamos sacar variables de grupo
            - % mujeres, %hombres
            - edad promedio del grupo
            - % de modalidades de trabajo en el grupo
            - desviacion estandar de la distancia del grupo a las oficinas
            - mediana del tiempo en la empresa del grupo
            - desviacion estandar del performance_score del grupo
        - Estadisticos del BI
            - % de personas que se van del grupo

- Leer las bases
    - Innovacion y originalidad
        - ¿Qué tan innovadora es la solución propuesta? 
        - ¿Se diferencia de los enfoques existentes para resolver el problema?
    - Complejidad tecnica
        - ¿El proyecto hace uso de técnicas avanzadas de análisis de datos y ciencia de datos? Esto podría incluir el uso de técnicas de aprendizaje automático, análisis estadístico avanzado, algoritmos de aprendizaje profundo, etc.
            - Utilizar upsampling no tiene sentido si no se entiende el negocio a profundidad. Otro motivo por el que no se uso es que los registros representa personas y no tiene sentido generar registros sinteticos.
            - No se utilizo downsampling dado que habian clases balanceadas
            - Graph feature engineering
                - La variable degree representa la cantidad de personas con las que ha tenido que interactuar el empleado durante su tiempo en la empresa. Se creo especialmente para los empleados de tipo jefe
    - Calidad del codigo
        - ¿Es el código limpio, organizado y bien documentado? 
        - ¿Se han seguido las mejores prácticas de codificación?
    - Defensa oral y calidad de la presentacion
        - ¿Qué tan claro es el equipo exponiendo sus ideas?
        - ¿Se han utilizado elementos gráficos eficaces y se han comunicado los resultados de manera clara y convincente? 
        - ¿La demo ha sido efectiva y juntamente con el speech articulan claramente las fortalezas de la solución implementada?
    - Trabajo en equipo
        - ¿Cómo se ha coordinado el equipo en la presentación y durante el desarrollo de la competencia? 
        - ¿Han demostrado una buena colaboración y comunicación efectiva?
    - Impacto positivo 
        - ¿Tiene el proyecto el potencial de generar un impacto positivo significativo en su área de aplicación?
            - Si usando estas predicciones se puede generar un BI con indicadores de abandono (variables influyentes) para anticiparse a la salida de un empleado, no solo en funcion a si mismo sino en funcion a su equipo.
            - Tener una mirada del equipo, mas alla de la mirada individual ayuda a entender el comportamiento de los empleados ya que se pueden correlacionar variables y hacer un analisis mas profundo.

##  `Tabla de contenidos:`
1. Contexto y desafío
2. Dataset
3. Metodología utilizada
4. Resultados del modelo
5. Impacto potencial


## `Links`:
- https://www.linkedin.com/pulse/xgboost-el-caballo-de-batalla-la-ciencia-datos-para-pompas-guti%C3%A9rrez/?originalSubdomain=es
- https://docs.google.com/presentation/d/1ARxzjrVAdPCQSNYK_WXcIxkQLTTXKQm0cYVFpNOWXxY/edit?usp=sharing

