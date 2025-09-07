import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import obspy
from obspy import read, UTCDateTime
import seisbench.models as sbm
import gc 

"""
Notas de las pruebas realizadas para tener en cuenta al momento de querer ejecutar este codigo 
- Es necesario contar y activar el entorno virtual necesario para ejecutar este código, principlamente con las dependencias de ObsPy SeisBench y flask
- Las pruebas se realizaron en la siguiente máquina:
    - Procesador: Ryzen 7 5700X
    - Memoria RAM: 32 GB DDR4 3200 MHz
    - GPU: Nvidia 4060 de 8GB
    - Disco Duro: NVMe de 1TB
El codigo contiene fuertes optimizaciones sobre el uso de memoria RAM, dado que en las pruebas realizadas, al querer procesar archivos mseed de 24 horas sobre una unica ventana de tiempo
(1440 minutos), el uso de memoria RAM se disparaba a mas de 32 GB, haciendo que el sistema se volviera inestable y se reiniciara el sistema operativo por falta de memoria RAM. Actualmente, el código ya
soporta 24 horas en un archivo (o multiples archivos) con ventanas de 1440 minutos, sin embargo, se recomienda que los archivos sean de 24 horas (como fue solicitado) pero la longitud de la ventana de tiempo sea de 
menos de maximo 72o minutos para evitar problemas de memoria RAM. En el caso de que se quiera procesar archivos de 24 horas con ventanas de 1440 minutos, se recomienda tener otros progemas cerrados, como navegadores web, editores de texto, etc.
ya que en la prueba, con mseed de 24 horas y una ventana de 1440 minutos, el uso de memoria RAM se disparaba a mas de 31 GB, teniendo solo 1 GB de margen para que el sistema operativo no se volviera inestable.

EL dataset recomendado y el que se encuentra por defecto en la totalidad del codigo es stead, sin embargo, el analisis fue realizado en totalidad por una persona que no es experta en sismos,
por lo que se recomienda que un experto en el tema realice un analsis sobre los resultados obtenidos por el modelo de detección de fases de SeisBench

La interfaz de usuario fue diseñada para ser intuitiva y fácil de usar, permitiendo a los usuarios cargar archivos MiniSEED, seleccionar filtros, y ejecutar modelos de detección de 
fases con un mínimo de configuración, excelente para usuarios expertos en sismos pero no expertos en programación.

Todo el codigo fue hecho gracias a la ayuda de la documentacion de SeisBench y ObsPy
"""

def load_mseed_file(filepath):
    """
    Carga un archivo MiniSEED desde la ruta especificada y lo convierte en un objeto Stream de ObsPy.
    Realiza una unión de las trazas dentro del stream, interpolando los datos en caso de solapamientos
    o gaps para asegurar la continuidad temporal de la señal.

    Esta función es crucial para preparar los datos sísmicos en un formato utilizable por los modelos
    de detección de fases de SeisBench y para su posterior procesamiento y visualización.

    Args:
        filepath (str): La ruta completa al archivo MiniSEED (.mseed, .ms, etc.) a cargar.

    Returns:
        obspy.core.stream.Stream or None:
            Un objeto `Stream` de ObsPy que contiene las trazas sísmicas del archivo cargado,
            con todas las trazas de un mismo canal y red unidas temporalmente.
            Retorna `None` si ocurre un error durante la lectura del archivo.

    Raises:
        (No levanta excepciones directamente, las captura e imprime un mensaje de error.)

    Notas:
        - Utiliza `obspy.read()` para la carga inicial del archivo. Este método es robusto
          y maneja varios formatos de datos sísmicos, siendo MiniSEED el más común para datos continuos.
        - La función `stream.merge(method=1, fill_value='interpolate')` es fundamental aquí:
            - `method=1` (o `'interpolation'`) asegura que si hay trazas contiguas o solapadas
              para el mismo canal (identificado por Network, Station, Location, Channel, Starttime),
              estas se unan en una única traza. Los gaps (huecos en el tiempo) o solapamientos
              entre trazas se resuelven interpolando los datos.
            - `fill_value='interpolate'` especifica que los valores en los gaps o solapamientos
              se deben rellenar mediante interpolación lineal, lo cual es útil para mantener
              la continuidad de la señal para algoritmos que requieren secuencias de tiempo completas.
            - Es importante que el stream resultante sea continuo para la inferencia de los modelos
              de SeisBench, ya que estos modelos esperan una secuencia temporal ininterrumpida de datos.
        - Los errores de lectura del archivo son capturados e informados sin detener la ejecución del script.
        - En la práctica, probé con varios archivos Mseed y los resultados fueron ligeramente superior cuando la traza de los sismos
          eran continuas, es decir, cuando no había gaps o solapamientos. Recomiendo que los archivos Mseed que se procesen sean continuos
          y que no tengan gaps o solapamientos significativos. Esto puede mejorar la precisión de los modelos de detección de fases.
    """
    try:
        # Intenta leer el archivo MiniSEED. ObsPy es capaz de detectar automáticamente el formato
        # del archivo.
        stream = read(filepath)
    except Exception as e:
        # Captura cualquier excepción que ocurra durante la lectura del archivo,
        # como archivos corruptos o inexistentes.
        print(f"Error al leer {filepath}: {e}")
        return None
    
    # Fusiona todas las trazas dentro del stream que compartan la misma identidad
    # (red, estación, localización, canal) y que sean temporalmente contiguas o solapadas.
    # El método 1 ('interpolate') se usa para rellenar posibles gaps o resolver solapamientos
    # interpolando los datos, asegurando un stream continuo.
    stream.merge(method=1, fill_value='interpolate')
    
    return stream

def apply_filter(stream, filter_params):
    """
    Aplica un filtro pasa-banda (bandpass) a una copia del objeto `Stream` de ObsPy.
    Esta función está diseñada con un enfoque en la optimización del uso de memoria,
    realizando una copia explícita del stream original y forzando la recolección de basura
    después de la operación.)

    El filtrado pasa-banda es una operación común en el procesamiento de señales sísmicas
    para aislar rangos de frecuencia de interés y reducir el ruido no deseado, lo cual
    puede mejorar el rendimiento de los modelos de detección de fases.

    El aplicar filtros fue una solución a la falta de poder hacer un fine tuning y un dataset personalizado alimentado con sismos de 
    México. Esto se debe a la falta de la infreestructura de computo necesaria para entrenar un modelo de detección de fases con un dataset personalizado. 
    Al momento de escribir este código, se contaba unicamanete con un GPU Nvidia T400 de 4GB, lo cual no es suficiente para entrenar un modelo de detección de fases con un dataset personalizado.


    Args:
        stream (obspy.core.stream.Stream):
            El objeto `Stream` de ObsPy al que se le aplicará el filtro. Se opera sobre una copia
            para no modificar el stream original.
        filter_params (dict):
            Un diccionario que contiene los parámetros para el filtro. Debe incluir las siguientes claves:
            - 'type' (str): Un identificador para el tipo de filtro (ej., "0.5-2Hz").
                            Este valor se almacena en los metadatos de la traza.
            - 'freqmin' (float): La frecuencia mínima (en Hz) del rango pasa-banda.
            - 'freqmax' (float): La frecuencia máxima (en Hz) del rango pasa-banda.

    Returns:
        obspy.core.stream.Stream or None:
            Un nuevo objeto `Stream` de ObsPy con el filtro pasa-banda aplicado a todas sus trazas.
            Las trazas dentro de este nuevo stream contendrán metadatos adicionales (`filter_type`,
            `freqmin`, `freqmax`) que describen el filtro aplicado.
            Retorna `None` si ocurre un error durante la aplicación del filtro.

    Raises:
        (No levanta excepciones directamente, las captura e imprime un mensaje de error.)

    Notas:
        - `stream.copy()`: Se utiliza para crear una copia profunda del stream original. Esto es crucial
          para asegurar que la operación de filtrado no modifique los datos del stream original,
          lo cual es importante si el stream original se va a utilizar para otros propósitos (ej.,
          procesamiento de la señal "original" sin filtrar).
        - `filtered_stream.filter(...)`: Este es el método de ObsPy que aplica el filtro.
            - `type='bandpass'`: Especifica un filtro pasa-banda.
            - `freqmin` y `freqmax`: Definen el rango de frecuencias que serán pasadas.
            - `corners=4`: Indica el número de esquinas (o polos) del filtro. Un mayor número
              generalmente resulta en una transición más pronunciada entre las bandas de paso y rechazo,
              pero también puede introducir más distorsión. Cuatro esquinas es un valor común.
            - `zerophase=True`: Aplica el filtro en ambas direcciones (adelante y atrás) para eliminar
              el desplazamiento de fase que introduce un filtro causal. Esto es deseable para el análisis
              sísmico ya que mantiene la posición temporal de las fases.
        - **Optimización de Memoria (`gc.collect()`):** El uso de `gc.collect()` explícitamente después
          de las operaciones de filtrado y en el bloque `except` es una medida agresiva para forzar
          la liberación de memoria no referenciada. Aunque Python tiene su propio recolector de basura
          automático, en aplicaciones que manejan grandes volúmenes de datos (como streams sísmicos)
          y donde el uso de memoria es crítico, invocar `gc.collect()` puede ayudar a liberar memoria
          más rápidamente, especialmente después de crear y descartar objetos grandes temporalmente.
        - **Metadatos del Filtro:** La adición de `filter_type`, `freqmin`, y `freqmax` a `tr.stats`
          es una buena práctica para mantener un registro de las operaciones de procesamiento
          realizadas en cada traza, lo cual es útil para la trazabilidad y depuración.
    """
    # Crear una copia profunda del stream original. Esto es esencial para evitar
    # modificar el stream de entrada directamente y para permitir el procesamiento
    # del stream original y múltiples versiones filtradas de forma independiente.
    filtered_stream = stream.copy()

    try:
        # Aplica un filtro pasa-banda a todas las trazas dentro del stream copiado.
        # 'corners=4' se refiere al orden del filtro (más esquinas = pendiente más pronunciada).
        # 'zerophase=True' aplica el filtro dos veces (adelante y atrás) para evitar
        # el desplazamiento de fase introducido por el filtrado.
        filtered_stream.filter(
            type='bandpass',
            freqmin=filter_params['freqmin'],
            freqmax=filter_params['freqmax'],
            corners=4,
            zerophase=True
        )

        # Itera sobre cada traza en el stream filtrado y añade metadatos
        # que describen el filtro aplicado. Esto es útil para la documentación
        # interna de la traza y para la trazabilidad.
        for tr in filtered_stream:
            tr.stats.filter_type = filter_params['type']
            tr.stats.freqmin = filter_params['freqmin']
            tr.stats.freqmax = filter_params['freqmax']

        # Fuerza la recolección de basura. Esto es una optimización de memoria
        # que intenta liberar inmediatamente la memoria de objetos que ya no están
        # siendo referenciados, lo cual es útil después de operaciones intensivas
        # con grandes estructuras de datos como los streams de ObsPy.
        gc.collect()

        return filtered_stream

    except Exception as e:
        # Captura cualquier error que pueda ocurrir durante la aplicación del filtro.
        print(f"Error al aplicar filtro {filter_params['type']}: {e}")
        # Si 'filtered_stream' ya fue creado antes del error, se intenta eliminar
        # explícitamente para ayudar a liberar memoria.
        if 'filtered_stream' in locals():
            del filtered_stream
        # Fuerza la recolección de basura nuevamente en caso de error para asegurar
        # la limpieza de cualquier objeto parcial o temporal.
        gc.collect()
        return None

def save_detailed_picks_to_csv(picks, model_name, basename, results_folder, filter_type="original"):
    """
    Guarda la información detallada de los 'picks' (detecciones de fases P y S) generados por
    un modelo de SeisBench en un archivo CSV. Cada fila del CSV representa un 'pick' individual,
    incluyendo metadatos relevantes como el tiempo de ocurrencia, el canal sísmico, la fase detectada,
    y el tipo de filtro aplicado a la señal original.

    Esta función es crucial para la persistencia de los resultados de la detección de fases,
    permitiendo un análisis posterior, la creación de catálogos de eventos o la evaluación
    del rendimiento de los modelos bajo diferentes condiciones de filtrado.

    Args:
        picks (list):
            Una lista de objetos 'Pick' (generalmente instancias de `seisbench.util.Pick`
            o clases similares que contienen atributos como `trace_id`, `phase`, `start_time`,
            `end_time`, `peak_time`, `peak_value`, `trace_length`). Estos objetos son
            la salida estándar de los métodos `classify` de los modelos de SeisBench.
        model_name (str):
            El nombre del modelo de IA que generó los picks (ej., "PhaseNet", "EQTransformer", "GPD").
            Se utiliza para nombrar el archivo CSV y para identificar la fuente de los picks.
        basename (str):
            El nombre base del archivo MiniSEED original que fue procesado (sin extensión).
            Se utiliza para nombrar el archivo CSV y para identificar el archivo de origen de los datos.
        results_folder (str):
            La ruta al directorio donde se guardará el archivo CSV.
        filter_type (str, optional):
            Una cadena que describe el tipo de filtro aplicado a la señal antes de la detección
            de los picks (ej., "original", "0.5-2Hz", "1-15Hz"). Este campo se añade al CSV
            para facilitar el análisis comparativo de los resultados de detección bajo diferentes
            regímenes de filtrado. Por defecto es "original".

    Returns:
        None: La función no retorna ningún valor, pero guarda un archivo CSV y imprime un mensaje
              de confirmación en la consola.

    Raises:
        IOError: Si hay problemas al escribir el archivo CSV (ej., permisos insuficientes).
        (Otras excepciones si los atributos de 'pick' no son accesibles, aunque se usan `getattr`
         para manejar esto con valores por defecto 'N/A'.)

    Notas:
        - **Formato del nombre del archivo:** El nombre del archivo CSV sigue el patrón
          `{basename}_{filter_type}_{model_name}_picks.csv`, lo que permite una identificación
          clara del contenido del archivo y facilita la organización de los resultados.
        - **Uso de `csv.writer`:** Se utiliza el módulo estándar `csv` de Python para escribir
          el archivo. `newline=''` es importante para evitar que se inserten filas en blanco
          adicionales entre las filas en Windows.
        - **Encabezados claros:** Los encabezados del CSV están diseñados para ser informativos
          y completos, incluyendo detalles sobre el archivo original, el modelo, el tipo de filtro,
          y las propiedades de cada pick.
        - **Acceso seguro a atributos (`getattr`):** Se utiliza `getattr(object, attribute, default_value)`
          para acceder a los atributos de cada objeto `pick`. Esto asegura que, si un atributo
          esperado no existe en un objeto `pick` por alguna razón, se utilice un valor por defecto
          ("N/A"), evitando errores y haciendo la función más robusta a variaciones menores
          en la estructura de los objetos `pick`.
        - **Formato de tiempo ISO 8601:** Los tiempos (`start_time`, `end_time`, `peak_time`)
          se convierten a cadenas en formato ISO 8601 utilizando `.isoformat()`. Este es un
          formato estándar e interoperable para la representación de fechas y horas, lo que
          facilita la lectura y el procesamiento por otras herramientas o lenguajes.
          `obspy.UTCDateTime` objetos ya tienen este método.
        - **Integración con SeisBench:** Esta función está diseñada para trabajar directamente
          con los objetos `Pick` que son el resultado del método `.classify()` de los modelos
          de SeisBench como `PhaseNet`, `EQTransformer`, y `GPD`.
    """
    # Construye la ruta completa del archivo CSV, incorporando el nombre base del archivo
    # MiniSEED, el tipo de filtro aplicado y el nombre del modelo. Esto asegura nombres
    # de archivo únicos e informativos.
    csv_filename = os.path.join(results_folder, f"{basename}_{filter_type}_{model_name}_picks.csv")

    # Abre el archivo CSV en modo escritura ('w') con soporte para nuevas líneas universal.
    with open(csv_filename, 'w', newline='') as csvfile:
        # Crea un objeto `csv.writer` para escribir filas de datos.
        csv_writer = csv.writer(csvfile)
        
        # Escribe la fila de encabezados para el archivo CSV. Estos encabezados
        # proporcionan una descripción clara de cada columna de datos.
        csv_writer.writerow([
            "filename", "modelo", "filter_type", "channel", "phase",
            "start_time", "end_time", "peak_time",
            "peak_value", "trace_length"
        ])

        # Itera sobre cada objeto 'pick' en la lista proporcionada.
        for pick in picks:
            # Obtiene la fase detectada (P, S, etc.) de manera segura, usando 'N/A' si no está presente.
            phase = getattr(pick, 'phase', 'N/A')

            # Obtiene los atributos de tiempo de manera segura. Si el atributo no existe o es None,
            # se asigna None para el procesamiento posterior.
            start_time = getattr(pick, 'start_time', None)
            end_time = getattr(pick, 'end_time', None)
            peak_time = getattr(pick, 'peak_time', None)

            # Convierte los objetos de tiempo (UTCDateTime de ObsPy) a formato de cadena ISO 8601.
            # Si el tiempo es None, se asigna "N/A". Esto asegura un formato de tiempo consistente.
            start_time_str = start_time.isoformat() if start_time else "N/A"
            end_time_str = end_time.isoformat() if end_time else "N/A"
            peak_time_str = peak_time.isoformat() if peak_time else "N/A"

            # Escribe una fila de datos en el archivo CSV con la información extraída de cada pick.
            csv_writer.writerow([
                basename,                                   # Nombre base del archivo original
                model_name,                                 # Nombre del modelo de IA
                filter_type,                                # Tipo de filtro aplicado
                getattr(pick, 'trace_id', 'N/A'),           # Identificador de la traza
                phase,                                      # Fase detectada ('P', 'S')
                start_time_str,                             # Tiempo de inicio del pick
                end_time_str,                               # Tiempo de fin del pick
                peak_time_str,                              # Tiempo del valor máximo dentro del pick
                getattr(pick, 'peak_value', 'N/A'),         # Valor máximo de probabilidad o amplitud en el pick
                getattr(pick, 'trace_length', 'N/A')        # Longitud de la traza asociada al pick
            ])

    # Imprime un mensaje de confirmación una vez que el archivo ha sido guardado.
    print(f"Guardado archivo de picks para {model_name} con filtro {filter_type}: {csv_filename}")


def save_eqt_detections_to_csv(detections, basename, results_folder, filter_type="original"):
    """
    Guarda las detecciones de terremotos generadas por el modelo EQTransformer en un archivo CSV.
    Esta función está diseñada específicamente para manejar la estructura de los objetos de detección
    devueltos por EQTransformer, que pueden incluir información adicional como la duración del evento.
    Se añade información sobre el tipo de filtro aplicado a la señal sísmica para permitir un análisis
    comparativo de las detecciones bajo diferentes condiciones de preprocesamiento.

    Args:
        detections (list):
            Una lista de objetos de detección. Idealmente, estos son objetos `seisbench.util.Detection`
            (o clases con atributos similares como `trace_id`, `start_time`, `end_time`, `peak_time`,
            `peak_value`). Estos objetos son la salida del método `classify` de EQTransformer
            cuando se generan detecciones de eventos.
        basename (str):
            El nombre base del archivo MiniSEED original que fue procesado (sin extensión).
            Se utiliza para nombrar el archivo CSV y para identificar el archivo de origen de los datos.
        results_folder (str):
            La ruta al directorio donde se guardará el archivo CSV.
        filter_type (str, optional):
            Una cadena que describe el tipo de filtro aplicado a la señal antes de la detección
            (ej., "original", "0.5-2Hz", "1-15Hz"). Este campo se añade al CSV
            para facilitar el análisis comparativo. Por defecto es "original".

    Returns:
        None: La función no retorna ningún valor, pero guarda un archivo CSV y imprime un mensaje
              de confirmación en la consola.

    Raises:
        IOError: Si hay problemas al escribir el archivo CSV (ej., permisos insuficientes).

    Noteas:
        - **Nombre del archivo:** El archivo CSV se nombra siguiendo el patrón
          `{basename}_{filter_type}_EQTransformer_detections.csv`, lo que facilita
          la identificación de los resultados de detección específicos de EQTransformer
          y el tipo de filtro aplicado.
        - **Estructura del CSV:** Incluye columnas para el nombre del archivo de origen,
          el tipo de filtro, el ID de la traza, los tiempos de inicio, fin y pico de la detección,
          el valor máximo de la detección, y la duración calculada del evento.
        - **Manejo de tiempos (`getattr`, `isoformat`, `UTCDateTime`):**
            - Se utiliza `getattr(detection, attribute, None)` para acceder de forma segura
              a los atributos de tiempo de los objetos `detection`, asignando `None` si el
              atributo no existe.
            - Los objetos de tiempo (que deberían ser `obspy.UTCDateTime` o compatibles)
              se convierten a cadenas en formato ISO 8601 usando `.isoformat()` si el
              método está disponible. Si no, se recurre a `str()`.
            - La duración se calcula restando `UTCDateTime` objetos. Para asegurar la
              compatibilidad, se verifica si `start_time` y `end_time` son instancias de
              `UTCDateTime` y se convierten si es necesario antes de calcular la duración.
              Los errores durante el cálculo de la duración se capturan e informan.
        - **`EQTransformer` Detecciones vs. Picks:** Es importante notar que EQTransformer
          puede generar tanto 'picks' (detecciones de fases P y S individuales) como
          'detections' (segmentos de tiempo donde se detecta la presencia de un evento sísmico,
          posiblemente conteniendo múltiples picks). Esta función se encarga de las 'detections',
          mientras que `save_detailed_picks_to_csv` (definida previamente) maneja los 'picks'.
    """
    # Construye la ruta completa del archivo CSV para las detecciones de EQTransformer.
    csv_filename = os.path.join(results_folder, f"{basename}_{filter_type}_EQTransformer_detections.csv")

    # Abre el archivo CSV en modo escritura, asegurando el manejo correcto de las nuevas líneas.
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Escribe los encabezados de las columnas del CSV.
        csv_writer.writerow([
            "filename", "filter_type", "trace_id", "start_time", "end_time",
            "peak_value", "peak_time", "duration"
        ])

        # Itera sobre cada objeto de detección proporcionado.
        for detection in detections:
            # Accede de forma segura a los atributos de tiempo de la detección.
            start_time = getattr(detection, 'start_time', None)
            end_time = getattr(detection, 'end_time', None)
            peak_time = getattr(detection, 'peak_time', None)

            # Convierte los objetos de tiempo a cadenas en formato ISO 8601.
            # Si el objeto de tiempo no tiene el método isoformat (por ejemplo, si es None o un tipo diferente),
            # se convierte a una cadena genérica.
            start_time_str = start_time.isoformat() if hasattr(start_time, 'isoformat') else str(start_time)
            end_time_str = end_time.isoformat() if hasattr(end_time, 'isoformat') else str(end_time)
            peak_time_str = peak_time.isoformat() if hasattr(peak_time, 'isoformat') else str(peak_time)

            # Inicializa la duración como "N/A".
            duration = "N/A"
            # Si tanto el tiempo de inicio como el de fin están presentes, calcula la duración.
            if start_time and end_time:
                try:
                    # Asegura que los objetos de tiempo sean de tipo UTCDateTime para realizar la resta.
                    if not isinstance(start_time, UTCDateTime):
                        start_time = UTCDateTime(start_time)
                    if not isinstance(end_time, UTCDateTime):
                        end_time = UTCDateTime(end_time)
                    duration = (end_time - start_time) # La resta de UTCDateTime devuelve segundos
                except Exception as e:
                    print(f"Error calculando duración: {e}")

            # Escribe la fila de datos en el archivo CSV.
            csv_writer.writerow([
                basename,                                   # Nombre base del archivo original
                filter_type,                                # Tipo de filtro aplicado
                getattr(detection, 'trace_id', 'N/A'),      # Identificador de la traza
                start_time_str,                             # Tiempo de inicio de la detección
                end_time_str,                               # Tiempo de fin de la detección
                getattr(detection, 'peak_value', 'N/A'),    # Valor máximo asociado a la detección
                peak_time_str,                              # Tiempo del valor máximo
                duration                                    # Duración de la detección
            ])

    # Imprime un mensaje de confirmación.
    print(f"Guardado archivo de detecciones de terremotos con filtro {filter_type}: {csv_filename}")

def process_stream_with_models(stream, pn_model, eqt_model, gpd_model, basename, results_folder, filter_type="original"):
    """
    Procesa un objeto `Stream` de ObsPy utilizando tres modelos de IA pre-entrenados de SeisBench:
    PhaseNet, EQTransformer y GPD (Generalized Phase Detection). Esta función realiza la
    clasificación de fases sísmicas (P y S) y, en el caso de EQTransformer, también detecta eventos
    sísmicos. Los resultados (picks y detecciones) se guardan en archivos CSV para su análisis.
    Además, genera las anotaciones de probabilidad (predictions) de cada modelo para su posterior
    visualización o uso.

    Esta función es el núcleo del procesamiento de datos sísmicos, aplicando la inteligencia
    artificial para extraer información crítica de los sismogramas y prepararla para la siguiente
    etapa del flujo de trabajo.

    Args:
        stream (obspy.core.stream.Stream):
            El objeto `Stream` de ObsPy que contiene las trazas sísmicas a procesar.
            Se espera que este stream ya haya sido cargado y preprocesado (ej., unido y filtrado).
        pn_model (seisbench.models.PhaseNet):
            Una instancia cargada del modelo PhaseNet de SeisBench.
            Para más detalles: [PhaseNet en SeisBench]
        eqt_model (seisbench.models.EQTransformer):
            Una instancia cargada del modelo EQTransformer de SeisBench.
            Para más detalles: [EQTransformer en SeisBench]
        gpd_model (seisbench.models.GPD):
            Una instancia cargada del modelo GPD de SeisBench.
            Para más detalles: [GPD en SeisBench]
        basename (str):
            El nombre base del archivo MiniSEED original que está siendo procesado (sin extensión).
            Se utiliza para nombrar los archivos de salida CSV.
        results_folder (str):
            La ruta al directorio donde se guardarán los archivos CSV de picks y detecciones.
        filter_type (str, optional):
            Una cadena que describe el tipo de filtro aplicado al `stream` antes de pasarlo a esta función
            (ej., "original", "0.5-2Hz"). Se utiliza para identificar los archivos de resultados.
            Por defecto es "original".

    Returns:
        dict: Un diccionario que contiene los objetos `Stream` anotados con las predicciones de probabilidad
              de cada modelo. Estas predicciones pueden ser utilizadas para la visualización.
              - "pn_preds": `Stream` con predicciones de PhaseNet.
              - "eqt_preds": `Stream` con predicciones de EQTransformer.
              - "gpd_preds": `Stream` con predicciones de GPD.

    Raises:
        (No levanta excepciones directamente, las captura e imprime mensajes de error.)

    Notas:
        - **Modelos de SeisBench:** La función espera instancias de modelos pre-entrenados
          de la librería `seisbench.models`. Se recomienda cargar estos modelos una sola vez
          al inicio de la aplicación para evitar sobrecarga de memoria y tiempo.
        - **`model.classify(stream, ...)`:** Este método es fundamental. Ejecuta el modelo
          de inferencia sobre el `stream` de entrada y devuelve un objeto que contiene los
          'picks' (tiempos de llegada de fases P y S) y, en el caso de EQTransformer,
          también 'detections' (segmentos de tiempo donde se infiere la ocurrencia de un evento).
          - `P_threshold` y `S_threshold`: Son umbrales de confianza que determinan qué tan
            alta debe ser la probabilidad de una fase para que sea considerada un 'pick'.
            Estos valores son críticos y deben ser ajustados según el rendimiento deseado
            y las características del ruido/señal de los datos.
        - **Manejo de `eqt_detections`:** EQTransformer puede generar un objeto `detections`
          adicional que encapsula los eventos completos, no solo picks individuales. Se utiliza
          `getattr` para acceder a este atributo de forma segura, ya que no todos los modelos
          lo producen.
        - **Guardado de resultados:** Se utilizan las funciones `save_detailed_picks_to_csv`
          y `save_eqt_detections_to_csv` (definidas previamente) para persistir los resultados
          en formato CSV. Esto es vital para el análisis posterior y la construcción de catálogos.
        - **`model.annotate(stream)`:** Este método es diferente a `classify`. En lugar de devolver
          picks o detecciones discretas, `annotate` devuelve un nuevo objeto `Stream` donde cada
          traza de entrada tiene trazas adicionales que representan las predicciones continuas
          de probabilidad del modelo para las fases P, S y/o ruido. Estas "predicciones" son útiles
          para visualizar el comportamiento del modelo sobre la señal bruta.
        - **Gestión de memoria (`del` y `gc.collect()`):**
          Después de guardar los picks y antes de retornar, se liberan explícitamente las referencias
          a los objetos `outputs_pn`, `outputs_eqt`, `outputs_gpd` y `eqt_detections`. Esto, combinado
          con una llamada a `gc.collect()`, ayuda a forzar al recolector de basura de Python a liberar
          la memoria ocupada por estos objetos grandes, lo cual es importante en pipelines de
          procesamiento de datos sísmicos para evitar el consumo excesivo de RAM, especialmente
          cuando se procesan muchos archivos o streams grandes.
    """
    print(f"Procesando stream con filtro: {filter_type}")

    # Ejecuta el método de clasificación de cada modelo sobre el stream de entrada.
    # Los umbrales (P_threshold, S_threshold) controlan la sensibilidad de la detección de fases.
    outputs_pn = pn_model.classify(stream)
    outputs_eqt = eqt_model.classify(stream, P_threshold=0.6, S_threshold=0.6)
    outputs_gpd = gpd_model.classify(stream, P_threshold=0.75, S_threshold=0.75)

    # Intenta obtener las detecciones de eventos de EQTransformer. EQTransformer es único
    # en que puede generar objetos de 'detección' de eventos además de 'picks' de fase.
    eqt_detections = []
    try:
        eqt_detections = getattr(outputs_eqt, 'detections', [])
    except Exception as e:
        print(f"Error al obtener detecciones de EQTransformer: {e}")

    # Imprime un resumen del número de picks y detecciones encontradas por cada modelo
    # para el tipo de filtro actual.
    print(f"{filter_type} - EQTransformer Picks: {len(outputs_eqt.picks)}")
    print(f"{filter_type} - EQTransformer Detecciones: {len(eqt_detections)}")
    print(f"{filter_type} - PhaseNet Picks: {len(outputs_pn.picks)}")
    print(f"{filter_type} - GPD Picks: {len(outputs_gpd.picks)}")

    # Guarda los picks generados por cada modelo en archivos CSV separados.
    save_detailed_picks_to_csv(outputs_pn.picks, "PhaseNet", basename, results_folder, filter_type)
    save_detailed_picks_to_csv(outputs_eqt.picks, "EQTransformer", basename, results_folder, filter_type)
    save_detailed_picks_to_csv(outputs_gpd.picks, "GPD", basename, results_folder, filter_type)

    # Si EQTransformer generó detecciones de eventos, guárdalas en un CSV separado.
    if eqt_detections:
        save_eqt_detections_to_csv(eqt_detections, basename, results_folder, filter_type)

    # Anota el stream con las predicciones de probabilidad continuas de cada modelo.
    # Estas predicciones son útiles para la visualización de la salida del modelo.
    pn_preds = pn_model.annotate(stream)
    eqt_preds = eqt_model.annotate(stream)
    gpd_preds = gpd_model.annotate(stream)

    # Libera las referencias a los objetos grandes que ya no se necesitan,
    # para ayudar a la gestión de memoria.
    del outputs_pn
    del outputs_eqt
    del outputs_gpd
    del eqt_detections
    # Fuerza al recolector de basura de Python a liberar la memoria de inmediato.
    gc.collect()

    # Retorna un diccionario con los streams anotados con las predicciones.
    return {
        "pn_preds": pn_preds,
        "eqt_preds": eqt_preds,
        "gpd_preds": gpd_preds
    }

def plot_filtered_streams_window(original_stream, filtered_streams, predictions_dict, t0, t1,
                                basename, window_index, results_img_folder):
    """
    Genera y guarda gráficos comparativos de la señal sísmica original y las señales filtradas,
    superponiendo las predicciones de probabilidad de los modelos de IA (PhaseNet, EQTransformer, GPD)
    para una ventana de tiempo específica. Esta función está fuertemente optimizada para el uso
    de memoria RAM, lo cual es crucial al procesar grandes volúmenes de datos.

    Cada gráfico muestra una traza sísmica junto con las curvas de probabilidad P, S y/o Detección
    generadas por los modelos, permitiendo una visualización directa del rendimiento de los modelos
    en diferentes rangos de frecuencia.

    Args:
        original_stream (obspy.core.stream.Stream):
            El objeto `Stream` de ObsPy que contiene las trazas sísmicas originales, sin filtrar.
        filtered_streams (dict):
            Un diccionario donde las claves son los tipos de filtro (ej., "0.5-2Hz", "1-15Hz")
            y los valores son los objetos `Stream` de ObsPy correspondientes con el filtro aplicado.
        predictions_dict (dict):
            Un diccionario anidado que contiene las predicciones de probabilidad de los modelos.
            La estructura esperada es:
            `predictions_dict[filter_type_or_original]['pn_preds']`
            `predictions_dict[filter_type_or_original]['eqt_preds']`
            `predictions_dict[filter_type_or_original]['gpd_preds']`
            donde `filter_type_or_original` es "original" o el nombre del filtro.
            Estos son los objetos `Stream` anotados devueltos por `process_stream_with_models`.
        t0 (obspy.core.utcdatetime.UTCDateTime):
            El tiempo de inicio UTC para la ventana de tiempo a graficar.
        t1 (obspy.core.utcdatetime.UTCDateTime):
            El tiempo de fin UTC para la ventana de tiempo a graficar.
        basename (str):
            El nombre base del archivo MiniSEED original. Se usa para nombrar el archivo de imagen.
        window_index (int):
            Un índice numérico para la ventana de tiempo actual, utilizado para nombrar el archivo
            de imagen y asegurar su unicidad.
        results_img_folder (str):
            La ruta al directorio donde se guardarán las imágenes generadas.

    Returns:
        None: La función no retorna ningún valor, pero guarda una imagen en formato PNG
              y realiza una limpieza agresiva de memoria.

    Raises:
        (No levanta excepciones directamente, las maneja internamente.)

    Notas:
        - **Recorte de Streams (`stream.slice(t0, t1)`):** Esta es la primera operación para
          cada stream (original y filtrado). Se extrae solo la porción de datos dentro de
          la ventana de tiempo `[t0, t1]`. Esto reduce significativamente la cantidad de
          datos a procesar y graficar, ahorrando memoria y tiempo.
        - **Estructura de Subplots:** La función crea una figura con múltiples subplots (`plt.subplots`).
          Para cada "señal" (original y cada versión filtrada), se reservan 4 filas: una para la
          traza sísmica y tres para las predicciones de probabilidad de los modelos PhaseNet,
          EQTransformer y GPD, respectivamente.
            - `sharex=True`: Asegura que todos los subplots compartan el mismo eje X (tiempo),
              lo que facilita la comparación visual de los eventos y predicciones a lo largo del tiempo.
            - `gridspec_kw={'hspace': 0.05}`: Reduce el espacio vertical entre los subplots,
              creando una visualización más compacta y continua.
        - **Gestión Agresiva de Memoria:**
            - `del variable`: Elimina explícitamente las referencias a objetos grandes
              (como `Stream`s, `Figure`, `Axes`) tan pronto como ya no son necesarios.
            - `gc.collect()`: Fuerza la ejecución del recolector de basura de Python
              para liberar la memoria ocupada por los objetos sin referencia.
            - `plt.close(fig)`, `plt.clf()`, `plt.cla()`: Estos comandos de Matplotlib
              cierran la figura actual, limpian la figura actual de todos los ejes, y
              limpian los ejes actuales, respectivamente. Son esenciales para liberar
              recursos gráficos asociados con la figura una vez que ha sido guardada,
              evitando fugas de memoria, especialmente en bucles donde se generan
              muchas figuras.
        - **`color_dict`:** Un diccionario simple para asignar colores consistentes a
          las diferentes fases (P, S, Detección) en los gráficos, mejorando la legibilidad.
        - **Guardado de la figura (`plt.savefig`):** La figura se guarda en formato PNG
          con una alta resolución (`dpi=150`) para asegurar la claridad visual. El nombre
          del archivo incluye el nombre base y el índice de la ventana.
    """
    # Recorta el stream original a la ventana de tiempo definida por t0 y t1.
    # Esto reduce la cantidad de datos a procesar y graficar, optimizando la memoria.
    subst_original = original_stream.slice(t0, t1)
    # Si el recorte no contiene datos, no hay nada que graficar, se limpia la referencia
    # y se retorna para ahorrar recursos.
    if len(subst_original) == 0:
        del subst_original
        gc.collect()
        return

    # Crea un diccionario para almacenar los streams filtrados y recortados.
    # Cada stream filtrado se recorta a la ventana de tiempo para optimizar.
    sliced_filtered_streams = {}
    for filter_type, filtered_stream in filtered_streams.items():
        sliced_stream = filtered_stream.slice(t0, t1)
        if len(sliced_stream) > 0:
            sliced_filtered_streams[filter_type] = sliced_stream

    # Determina el número de subplots necesarios.
    # +1 es para el stream original. Cada stream (original o filtrado) tendrá 4 filas
    # (señal + 3 modelos de predicción).
    n_filters = len(sliced_filtered_streams) + 1
    n_rows = n_filters * 4

    # Crea la figura y el conjunto de subplots.
    # sharex=True: Todos los subplots comparten el mismo eje X (tiempo), facilitando la comparación.
    # hspace=0.05: Reduce el espacio vertical entre los subplots para una vista más compacta.
    fig, axs = plt.subplots(n_rows, 1, figsize=(15, n_filters * 6),
                           sharex=True,
                           gridspec_kw={'hspace': 0.05})

    # Diccionario de colores para representar diferentes fases en los gráficos.
    color_dict = {"P": "C0", "S": "C1", "Detection": "C2"}

    # Procesa y grafica la señal original.
    # `current_row` lleva el control de la fila inicial para cada conjunto de subplots.
    current_row = 0
    process_stream_window(subst_original, "original", predictions_dict["original"],
                         t0, t1, current_row, axs, color_dict)
    current_row += 4 # Avanza 4 filas para el siguiente conjunto de subplots

    # Procesa y grafica cada señal filtrada.
    for filter_type, sliced_stream in sliced_filtered_streams.items():
        process_stream_window(sliced_stream, filter_type, predictions_dict[filter_type],
                             t0, t1, current_row, axs, color_dict)
        current_row += 4

        # Limpia explícitamente la referencia al stream recortado y fuerza la recolección
        # de basura para liberar memoria después de procesar cada stream filtrado.
        del sliced_stream
        gc.collect()

    # Crea un formateador de tiempo para el eje X, asegurando una visualización legible del tiempo.
    formatter = create_time_formatter(t0)
    for ax in axs:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))

    # Establece la etiqueta del eje X solo en el último subplot para evitar redundancia.
    axs[-1].set_xlabel("Tiempo (HH:MM:SS)")

    # Ajusta automáticamente los parámetros del subplot para un ajuste apretado,
    # evitando el solapamiento de etiquetas.
    plt.tight_layout()

    # Construye la ruta completa del archivo de imagen y guarda la figura.
    img_filename = os.path.join(results_img_folder, f"{basename}_comparison_window{window_index}.png")
    plt.savefig(img_filename, dpi=150) # Guarda con una resolución de 150 DPI

    # LIMPIEZA AGRESIVA DE MEMORIA DE MATPLOTLIB Y PYTHON
    # Cierra la figura para liberar los recursos gráficos asociados.
    plt.close(fig)
    # Limpia la figura actual de todos los ejes.
    plt.clf()
    # Limpia los ejes actuales.
    plt.cla()

    # Elimina todas las referencias a objetos grandes y fuerza la recolección de basura.
    del fig, axs, subst_original, sliced_filtered_streams, formatter
    gc.collect()

def process_stream_window(stream, filter_type, predictions, t0, t1, row_offset, axs, color_dict):
    """
    Procesa, normaliza y grafica una ventana de tiempo específica de un objeto `Stream` de ObsPy
    (ya sea original o filtrado) junto con las predicciones de probabilidad de los modelos de IA
    (PhaseNet, EQTransformer, GPD) en un conjunto de ejes proporcionado por `matplotlib`.
    La función está optimizada para el uso de memoria RAM, realizando operaciones de limpieza
    inmediata después de procesar cada subcomponente.

    Esta función es una pieza auxiliar clave para `plot_filtered_streams_window`, encargada
    de dibujar las trazas sísmicas y las curvas de probabilidad en sus respectivos subplots.

    Args:
        stream (obspy.core.stream.Stream):
            El objeto `Stream` de ObsPy a graficar para la ventana de tiempo actual.
            Este stream ya debería estar recortado a la ventana `[t0, t1]`.
        filter_type (str):
            Una cadena que describe el tipo de filtro aplicado al stream (ej., "original", "0.5-2Hz").
            Se utiliza para generar el título del subplot.
        predictions (dict):
            Un diccionario que contiene las predicciones de probabilidad de los modelos
            para el `stream` actual. La estructura esperada es:
            `predictions['pn_preds']`, `predictions['eqt_preds']`, `predictions['gpd_preds']`.
            Cada uno de estos valores es un `Stream` anotado con trazas de probabilidad.
        t0 (obspy.core.utcdatetime.UTCDateTime):
            El tiempo de inicio UTC de la ventana a graficar.
        t1 (obspy.core.utcdatetime.UTCDateTime):
            El tiempo de fin UTC de la ventana a graficar.
        row_offset (int):
            El índice de la fila inicial en el arreglo de ejes (`axs`) donde se comenzará
            a graficar este stream y sus predicciones.
        axs (numpy.ndarray):
            Un arreglo de objetos `Axes` de Matplotlib donde se realizarán los gráficos.
            Se espera que sea un arreglo 1D que representa las filas de subplots.
        color_dict (dict):
            Un diccionario de asignación de colores para las diferentes fases (ej., {"P": "C0", "S": "C1"}).

    Returns:
        None: La función dibuja en los ejes proporcionados pero no retorna ningún valor.

    Raises:
        (No levanta excepciones directamente, las maneja internamente.)

    Notas:
        - **Normalización de Amplitud:** Cada traza sísmica se normaliza por su valor absoluto máximo.
          Esto asegura que las trazas de diferentes canales o con diferentes rangos de amplitud
          puedan ser graficadas de manera efectiva en el mismo subplot sin que una domine a las demás.
        - **Offsets Verticales:** Las trazas dentro de un mismo subplot se grafican con un desplazamiento
          vertical (`offset`) para evitar que se superpongan, haciendo que cada canal sea claramente visible.
        - **Graficación por Trazas Individuales:** Para optimización de memoria, la función itera sobre
          cada `Trace` dentro de un `Stream` y la grafica individualmente, liberando referencias
          (`del norm_data`) lo antes posible.
        - **`current_ax`:** Se calcula el `current_ax` para cada modelo de predicción, asegurando
          que las probabilidades de PhaseNet, EQTransformer y GPD se grafiquen en sus propias
          filas de subplot (`row_offset + 1`, `row_offset + 2`, `row_offset + 3`).
        - **Coherencia Temporal (Offset de tiempo):** Se calcula un `offset` de tiempo (`subpreds[0].stats.starttime - stream[0].stats.starttime`)
          para asegurar que las predicciones se grafiquen correctamente alineadas en el tiempo
          con las trazas sísmicas a las que corresponden. Esto es importante si el tiempo de inicio
          de las predicciones no coincide exactamente con el de las trazas sísmicas.
        - **Limpieza de Memoria (`del`, `gc.collect()`):** Se realizan llamadas explícitas
          a `del` y `gc.collect()` después de procesar cada traza y cada conjunto de predicciones
          para liberar agresivamente la memoria RAM, lo cual es vital en entornos de procesamiento
          continuo o con muchos gráficos.
        - **Límites Y y X:** Se establecen límites adecuados para los ejes Y (amplitud/probabilidad)
          y X (tiempo) para asegurar una visualización clara. El límite Y para las probabilidades
          se establece de 0 a 1.1 para incluir la escala completa.
    """
    # Establece el título del primer subplot para la sección actual (original o filtrado).
    title = "Original" if filter_type == "original" else f"Filtro: {filter_type}"
    axs[row_offset].set_title(title, fontsize=12, fontweight='bold')

    # Define colores predeterminados para las trazas sísmicas (ej., Z, N, E).
    colors = ['k', 'r', 'b'] # Negro, Rojo, Azul
    offset_factor = 1.2 # Factor para desplazar verticalmente las trazas y evitar solapamiento

    # Procesa cada traza sísmica individualmente para optimizar el uso de memoria.
    for i, tr in enumerate(stream):
        color = colors[i % len(colors)] # Asigna un color rotatorio
        
        # Normaliza los datos de la traza para que estén en un rango consistente (-1 a 1).
        # Esto es crucial para graficar múltiples trazas con diferentes amplitudes en el mismo eje.
        max_abs = np.max(np.abs(tr.data))
        if max_abs > 0:  # Evita la división por cero si la traza es plana
            norm_data = tr.data / max_abs
        else:
            norm_data = tr.data
        
        # Calcula el desplazamiento vertical para esta traza.
        offset = (len(stream) - 1 - i) * offset_factor
        
        # Grafica la traza normalizada con su desplazamiento.
        axs[row_offset].plot(tr.times(), norm_data + offset, color=color, label=tr.stats.channel)

        # Libera la referencia a `norm_data` inmediatamente para gestionar la memoria.
        del norm_data

    # Configura los límites del eje Y para la traza sísmica y el eje X para la ventana de tiempo.
    axs[row_offset].set_ylim(-1, len(stream) * offset_factor) # Ajusta el límite Y según el número de trazas y offset
    axs[row_offset].set_xlim(0, t1 - t0) # El eje X va desde 0 hasta la duración de la ventana
    axs[row_offset].set_ylabel("Amplitud\nNormalizada") # Etiqueta del eje Y
    axs[row_offset].legend(loc="upper right") # Muestra la leyenda del canal

    # Define los modelos y sus correspondientes streams de predicciones para iterar.
    models_data = [
        ("EQTransformer", predictions["eqt_preds"]),
        ("PhaseNet", predictions["pn_preds"]),
        ("GPD", predictions["gpd_preds"])
    ]

    # Itera sobre cada modelo de predicción.
    for i, (model_name, preds_full) in enumerate(models_data):
        # Selecciona el subplot correcto para las predicciones del modelo actual.
        # Las predicciones se grafican en las filas siguientes a la traza sísmica.
        current_ax = axs[row_offset + i + 1]

        # Recorta el stream de predicciones completo a la ventana de tiempo actual.
        subpreds = preds_full.slice(t0, t1)
        
        # Si no hay predicciones en la ventana, configura el eje y continúa.
        if len(subpreds) == 0:
            current_ax.set_ylabel(model_name)
            current_ax.set_ylim(0, 1.1) # Rango de probabilidad de 0 a 1.1
            del subpreds # Libera referencia
            continue

        # Calcula el offset de tiempo para alinear las predicciones con la traza sísmica.
        # Esto es necesario si los tiempos de inicio del stream y las predicciones difieren ligeramente.
        offset = subpreds[0].stats.starttime - stream[0].stats.starttime

        # Grafica cada traza de predicción individualmente.
        for pred_trace in subpreds:
            try:
                # Intenta extraer el nombre del modelo y la clase de fase del nombre del canal.
                pred_model, pred_class = pred_trace.stats.channel.split("_")
            except Exception:
                # Si el formato no es el esperado, usa el canal completo y una clase vacía.
                pred_model = pred_trace.stats.channel
                pred_class = ""

            # Omite las trazas de ruido si el nombre de la clase es "N".
            if pred_class == "N":
                continue

            # Obtiene el color de la clase de fase del diccionario de colores.
            c = color_dict.get(pred_class, "C0") # "C0" es el color por defecto de matplotlib

            # Grafica la traza de predicción. Se suma el offset de tiempo para la alineación.
            current_ax.plot(offset + pred_trace.times(), pred_trace.data,
                          label=pred_class, color=c)

        # Configura las etiquetas y límites del eje para el subplot de predicciones.
        current_ax.set_ylabel(model_name)
        current_ax.legend(loc="upper right")
        current_ax.set_ylim(0, 1.1) # Las probabilidades van de 0 a 1

        # Limpia inmediatamente la referencia a las predicciones recortadas y fuerza la recolección de basura.
        del subpreds
        gc.collect()

def generate_individual_plots(original_stream, filtered_streams, predictions_dict, basename, results_img_folder, window_length_minutes):
    """
    Genera y guarda gráficos individuales para cada tipo de stream (original y cada uno de los filtrados)
    a lo largo de todas las ventanas de tiempo definidas. Cada gráfico muestra la traza sísmica
    junto con las predicciones de probabilidad de fase (P y S) de los modelos de IA (PhaseNet,
    EQTransformer, GPD). La función está diseñada con una fuerte optimización del uso de memoria
    RAM para manejar eficientemente grandes conjuntos de datos.

    Esta función proporciona una visualización detallada del comportamiento de los modelos
    para cada tipo de procesamiento de la señal, facilitando la inspección visual y la
    evaluación de las detecciones.

    Args:
        original_stream (obspy.core.stream.Stream):
            El objeto `Stream` de ObsPy que contiene las trazas sísmicas originales, sin filtrar.
        filtered_streams (dict):
            Un diccionario donde las claves son los tipos de filtro (ej., "0.5-2Hz")
            y los valores son los objetos `Stream` de ObsPy correspondientes con el filtro aplicado.
        predictions_dict (dict):
            Un diccionario anidado que contiene las predicciones de probabilidad de los modelos.
            La estructura esperada es:
            `predictions_dict[filter_type_or_original]['pn_preds']`
            `predictions_dict[filter_type_or_original]['eqt_preds']`
            `predictions_dict[filter_type_or_original]['gpd_preds']`
            Cada uno de estos valores es un `Stream` anotado con trazas de probabilidad.
        basename (str):
            El nombre base del archivo MiniSEED original que está siendo procesado (sin extensión).
            Se utiliza para nombrar los archivos de imagen.
        results_img_folder (str):
            La ruta al directorio base donde se guardarán las imágenes. Dentro de esta carpeta,
            se crearán subcarpetas para cada tipo de filtro.
        window_length_minutes (int):
            La duración de cada ventana de tiempo en minutos para la cual se generará un gráfico individual.

    Returns:
        None: La función no retorna ningún valor, pero guarda múltiples imágenes PNG
              y realiza una limpieza agresiva de memoria durante su ejecución.

    Raises:
        (No levanta excepciones directamente, las maneja internamente.)

    Notas:
        - **Definición de Ventanas:** La función calcula las ventanas de tiempo dividiendo la duración total
          del `original_stream` en segmentos de `window_length_minutes`.
        - **Estructura de Carpetas de Salida:** Para una mejor organización, se crea una subcarpeta
          dentro de `results_img_folder` para cada `filter_type` (ej., `results_img_folder/original/`,
          `results_img_folder/0.5-2Hz/`), donde se guardan los gráficos correspondientes.
        - **Bucles Anidados para Procesamiento:** La función utiliza bucles anidados:
            1. Un bucle externo itera a través del stream original y cada stream filtrado.
            2. Un bucle interno itera sobre las ventanas de tiempo dentro de cada stream.
        - **Creación Dinámica de Subplots (`gridspec_kw`):** Para cada ventana y tipo de filtro,
          se crea una nueva figura con 4 subplots: uno grande para la señal sísmica y tres más
          pequeños para las predicciones de PhaseNet, EQTransformer y GPD, respectivamente.
          `height_ratios=[2, 1, 1, 1]` asigna el doble de altura a la fila de la señal sísmica.
        - **Lógica de Graficación Duplicada/Integrada:** La lógica para graficar las trazas sísmicas
          normalizadas con offset y las trazas de probabilidad de los modelos es similar a la
          función `process_stream_window` pero está directamente integrada aquí. Esto se hace
          para maximizar la limpieza de memoria al procesar una ventana a la vez.
        - **Manejo de Trazas de Ruido (`pred_class == "N"`):** Las trazas de predicción que corresponden
          a "ruido" (`_N`) son explícitamente omitidas de la graficación para centrarse en las
          predicciones de fase P y S.
        - **`color_dict`:** Se utiliza un diccionario de colores para asegurar una representación
          visual consistente de las fases P y S. Se corrigió un posible error tipográfico de "S1" a "C1"
          para usar los colores cíclicos predeterminados de Matplotlib.
        - **Gestión Agresiva de Memoria (`del`, `gc.collect()`, `plt.close()`, `plt.clf()`, `plt.cla()`):**
          Estas operaciones se realizan en múltiples puntos clave:
            - Después de recortar `subst` si está vacío.
            - Después de procesar cada `subpreds` de un modelo individualmente.
            - Después de procesar cada `tr` de la señal sísmica.
            - Después de guardar cada figura (`plt.close(fig)`, `plt.clf()`, `plt.cla()`).
            - Al final de cada iteración del bucle externo (por cada tipo de filtro) para una limpieza
              adicional.
          Esta estrategia es vital para prevenir el agotamiento de la memoria al generar un gran
          número de gráficos.
        - **`create_time_formatter`:** Utiliza una función auxiliar para formatear los ticks
          del eje X a un formato de tiempo legible (HH:MM:SS).
    """
    # Convierte la duración de la ventana de minutos a segundos.
    wlength = window_length_minutes * 60
    # Obtiene los tiempos de inicio y fin del stream
    starttime = original_stream[0].stats.starttime
    endtime = original_stream[0].stats.endtime
    # Calcula la duración total del stream en segundos.
    total_seconds = int(endtime - starttime)

    # Diccionario de colores para cada fase detectada. "C0", "C1", "C2" son los colores
    # predeterminados de Matplotlib del ciclo de colores.
    color_dict = {"P": "C0", "S": "C1", "Detection": "C2"}

    # Itera sobre el stream original y luego sobre cada stream filtrado.
    # `[("original", original_stream)]` añade el stream original al inicio de la iteración.
    for filter_type, stream in [("original", original_stream)] + list(filtered_streams.items()):
        # Obtiene las predicciones correspondientes a este tipo de filtro.
        predictions = predictions_dict[filter_type]

        # Crea una carpeta específica para guardar las imágenes de este tipo de filtro.
        filter_img_folder = os.path.join(results_img_folder, filter_type)
        os.makedirs(filter_img_folder, exist_ok=True)

        window_index = 0 # Reinicia el índice de la ventana para cada tipo de filtro.

        # Itera sobre el stream en ventanas de `wlength` segundos.
        for s in range(0, total_seconds, wlength):
            t0 = starttime + s        # Tiempo de inicio de la ventana actual
            t1 = t0 + wlength         # Tiempo de fin de la ventana actual
            
            # Recorta el stream a la ventana de tiempo actual.
            subst = stream.slice(t0, t1)
            # Si el recorte está vacío, libera la referencia y pasa a la siguiente ventana.
            if len(subst) == 0:
                del subst
                continue

            # Crea una nueva figura con 4 subplots.
            # sharex=True: todos los subplots comparten el mismo eje X.
            # height_ratios: asigna una mayor altura a la traza sísmica (ratio 2) y menor a las predicciones (ratio 1).
            fig, axs = plt.subplots(4, 1, figsize=(15, 7),
                                   sharex=True,
                                   gridspec_kw={'hspace': 0.05, 'height_ratios': [2, 1, 1, 1]})

            # Establece el título principal de la figura.
            fig.suptitle(f"{filter_type}", fontsize=14, fontweight='bold')

            # Define la lista de modelos de predicción a procesar.
            models_to_process = [
                ("EQTransformer", predictions["eqt_preds"]),
                ("PhaseNet", predictions["pn_preds"]),
                ("GPD", predictions["gpd_preds"])
            ]

            # Itera sobre cada modelo y grafica sus predicciones.
            for i, (model_name, preds_full) in enumerate(models_to_process):
                # Recorta las predicciones completas a la ventana de tiempo actual.
                subpreds = preds_full.slice(t0, t1)
                
                # Si no hay predicciones en la ventana, configura el eje y limpia.
                if len(subpreds) == 0:
                    axs[i+1].set_ylabel(model_name)
                    axs[i+1].set_ylim(0, 1.1)
                    del subpreds
                    continue

                # Calcula el offset de tiempo para alinear las predicciones correctamente.
                offset = subpreds[0].stats.starttime - subst[0].stats.starttime

                # Grafica cada traza de predicción individualmente.
                for pred_trace in subpreds:
                    try:
                        # Extrae el modelo y la clase (P, S, N) del nombre del canal.
                        pred_model, pred_class = pred_trace.stats.channel.split("_")
                    except Exception:
                        pred_model = pred_trace.stats.channel
                        pred_class = ""

                    # Omite las trazas de ruido.
                    if pred_class == "N":
                        continue

                    # Obtiene el color correspondiente a la clase de fase.
                    c = color_dict.get(pred_class, "C0")
                    
                    # Grafica la predicción.
                    axs[i+1].plot(offset + pred_trace.times(), pred_trace.data,
                                 label=pred_class, color=c)

                # Configura las etiquetas y límites para el subplot de predicciones.
                axs[i+1].set_ylabel(model_name)
                axs[i+1].legend(loc="upper right")
                axs[i+1].set_ylim(0, 1.1)

                # Limpia inmediatamente las referencias a las predicciones recortadas y fuerza la recolección de basura.
                del subpreds
                gc.collect()

            # --- Graficación de las trazas sísmicas ---
            # Define colores y un factor de desplazamiento para las trazas sísmicas.
            colors = ['k', 'r', 'b'] # Negro, Rojo, Azul
            offset_factor = 1.2

            # Itera sobre cada traza sísmica en el stream recortado.
            for i, tr in enumerate(subst):
                color = colors[i % len(colors)] # Asigna un color rotatorio
                
                # Normaliza los datos de la traza para graficar.
                max_abs = np.max(np.abs(tr.data))
                if max_abs > 0:
                    norm_data = tr.data / max_abs
                else:
                    norm_data = tr.data
                
                # Calcula el desplazamiento vertical.
                offset = (len(subst) - 1 - i) * offset_factor
                
                # Grafica la traza normalizada.
                axs[0].plot(tr.times(), norm_data + offset, color=color, label=tr.stats.channel)
                del norm_data  # Libera la referencia inmediatamente

            # Configura los límites y etiquetas para el subplot de la señal sísmica.
            axs[0].set_ylim(-1, len(subst) * offset_factor)
            axs[0].set_xlim(0, wlength) # El eje X va desde 0 hasta la duración de la ventana
            axs[0].set_ylabel("Amplitud\nNormalizada")
            axs[0].legend(loc="upper right")

            # --- Formato del eje de tiempo ---
            # Crea un formateador para el eje X.
            formatter = create_time_formatter(t0)
            for ax in axs:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(formatter))

            # Establece la etiqueta del eje X solo en el último subplot.
            axs[3].set_xlabel("Tiempo (HH:MM:SS)")

            # --- Guardar figura y Limpieza ---
            # Construye la ruta para guardar la imagen.
            img_filename = os.path.join(filter_img_folder, f"{basename}_{filter_type}_window{window_index}.png")
            plt.savefig(img_filename, dpi=150) # Guarda con alta resolución

            # Realiza una limpieza agresiva de memoria de Matplotlib y Python.
            plt.close(fig) # Cierra la figura
            plt.clf()      # Limpia la figura actual
            plt.cla()      # Limpia los ejes actuales

            # Libera referencias explícitamente y fuerza la recolección de basura.
            del fig, axs, subst, formatter
            gc.collect()

            window_index += 1 # Incrementa el índice de la ventana

        # Limpieza adicional después de procesar todas las ventanas para un tipo de filtro.
        gc.collect()


def create_time_formatter(t0_ref):
    """
    Crea y retorna una función de formateo personalizada para el eje X (tiempo) de los gráficos
    de Matplotlib. Esta función es esencial para presentar los tiempos sísmicos de manera
    legible y relativa a un punto de inicio específico (`t0_ref`), mostrando solo la hora,
    minutos y segundos.

    Args:
        t0_ref (obspy.core.utcdatetime.UTCDateTime):
            El tiempo de referencia UTC (punto de inicio) para el eje X del gráfico.
            El formateador calculará el tiempo absoluto sumando el valor `x` (que
            representa segundos desde el inicio del eje) a este tiempo de referencia.

    Returns:
        function: Una función (`time_formatter`) que toma dos argumentos (`x`, `pos`)
                  y retorna una cadena de texto formateada como "HH:MM:SS".

    Notas:
        - **Cierre de Clousure:** La función `create_time_formatter` es una fábrica de funciones.
          Retorna una función interna (`time_formatter`) que "cierra" sobre el valor de `t0_ref`.
          Esto significa que `time_formatter` siempre tendrá acceso al `t0_ref` con el que fue creada,
          incluso después de que `create_time_formatter` haya terminado su ejecución.
        - **Argumentos de `time_formatter` (`x`, `pos`):**
            - `x`: Es el valor numérico de la posición del tick en el eje (generalmente en segundos
              relativos al inicio del eje, o 0).
            - `pos`: Es la posición de la marca, que a menudo no se usa para el formateo.
        - **Conversión a `UTCDateTime`:** El valor `x` se suma a `t0_ref` para obtener un objeto
          `UTCDateTime` que representa el tiempo absoluto en ese punto del eje.
        - **Formato `"%H:%M:%S"`:** El método `strftime` del objeto `UTCDateTime` se utiliza para
          formatear el tiempo como una cadena que muestra solo las horas, minutos y segundos.
          Este formato es conciso y relevante para la mayoría de las visualizaciones de ventanas
          sísmicas.
    """
    def time_formatter(x, pos):
        """
        Función interna que formatea un valor numérico de tiempo (en segundos relativos)
        a una cadena de tiempo HH:MM:SS, usando t0_ref como punto de inicio.
        """
        # Convierte el valor 'x' (segundos desde el inicio del gráfico) a un objeto UTCDateTime.
        time = t0_ref + x
        # Formatea el objeto UTCDateTime a una cadena HH:MM:SS.
        return time.strftime("%H:%M:%S")
    
    # Retorna la función interna que será usada por Matplotlib.
    return time_formatter

def process_file(filepath, pn_model, eqt_model, gpd_model, base_output_dir_for_file, window_length_minutes):
    """
    Orquesta el procesamiento completo de un archivo MiniSEED (`.mseed`).
    Esta función carga el archivo, aplica una serie de filtros de banda de paso
    predefinidos, pasa el stream original y cada stream filtrado a tres modelos
    de detección de fases sísmicas (PhaseNet, EQTransformer, GPD), guarda los
    resultados (picks y detecciones) en archivos CSV, y genera diversas visualizaciones
    (gráficos individuales y comparativos) de los datos sísmicos y las predicciones
    de los modelos.

    La función está fuertemente optimizada para la gestión de memoria RAM, procesando
    los streams y generando gráficos por ventanas de tiempo y liberando recursos
    agresivamente para manejar archivos de gran tamaño.

    Args:
        filepath (str):
            La ruta completa al archivo MiniSEED (`.mseed`) que se va a procesar.
        pn_model (seisbench.models.PhaseNet):
            Una instancia cargada del modelo PhaseNet de SeisBench.
        eqt_model (seisbench.models.EQTransformer):
            Una instancia cargada del modelo EQTransformer de SeisBench.
        gpd_model (seisbench.models.GPD):
            Una instancia cargada del modelo GPD de SeisBench.
        base_output_dir_for_file (str):
            El directorio base donde se crearán las subcarpetas para guardar
            todos los resultados (CSV e imágenes) relacionados con este archivo específico.
        window_length_minutes (int):
            La duración de las ventanas de tiempo en minutos que se utilizarán para
            generar los gráficos (tanto individuales como comparativos).

    Returns:
        None: La función no retorna ningún valor, pero genera múltiples archivos
              CSV y PNG en las carpetas de resultados.

    Raises:
        (Captura e imprime errores durante la carga del archivo o el procesamiento de streams.)

    Notas:
        - **Carga y Verificación (`load_mseed_file`):** Primero, se carga el archivo MiniSEED.
          Si hay un error o el archivo no se puede leer, la función termina tempranamente.
        - **Definición de Directorios de Salida:** Se crean subcarpetas específicas
          para las imágenes y los archivos CSV dentro del `base_output_dir_for_file`
          para una organización clara de los resultados de cada archivo.
        - **Filtros de Banda de Paso:** Se define una lista de parámetros de filtro
          (rangos de frecuencia). El procesamiento se realiza para la señal original
          y para cada una de estas versiones filtradas.
        - **Procesamiento del Stream (`process_stream_with_models`):** Esta es una
          función clave que:
            - Ejecuta los modelos de IA (`pn_model`, `eqt_model`, `gpd_model`)
              sobre el stream (original o filtrado).
            - Guarda los `picks` y `detections` (si aplica) en archivos CSV.
            - Retorna un diccionario con las predicciones de probabilidad de cada modelo
              (`predictions_dict`).
        - **Estrategia de Optimización de Memoria:**
            - **Procesamiento de Filtros Individualmente:** A diferencia de una estrategia
              donde todos los streams filtrados se mantienen en memoria simultáneamente,
              aquí cada filtro se aplica y procesa uno por uno. Solo se mantiene en
              `filtered_streams` la referencia necesaria para la graficación comparativa.
            - **Liberación Agresiva de Memoria (`del`, `gc.collect()`, `plt.close('all')`):**
              Después de cada fase de procesamiento (ej., después de aplicar un filtro
              y procesar con modelos, después de generar gráficos individuales, después
              de cada ventana de gráficos comparativos, y al final de la función),
              se liberan explícitamente las referencias a objetos grandes (streams,
              predicciones, figuras de Matplotlib) y se fuerza al recolector de basura
              a liberar la memoria. Esto es crucial para manejar grandes volúmenes de datos
              y evitar errores de "MemoryError".
        - **Generación de Gráficos:**
            - **`generate_individual_plots`:** Crea gráficos separados para cada tipo de filtro
              (original y los filtrados) en ventanas de tiempo. Estos gráficos se guardan
              en subcarpetas específicas del filtro.
            - **`plot_filtered_streams_window`:** Crea gráficos comparativos que muestran
              la señal original y *todas* las señales filtradas, junto con sus predicciones
              de modelo, para una ventana de tiempo específica. Estos se guardan en una
              carpeta de "comparison".
        - **`plt.close('all')`:** Al final de la función, esta llamada cierra todas las
          figuras de Matplotlib que puedan haber quedado abiertas, lo que es otra medida
          importante de limpieza de recursos gráficos.
    """
    print(f"Procesando {filepath}")
    # Carga el archivo mseed. Si hay un error al cargar, la función termina.
    original_stream = load_mseed_file(filepath)
    if original_stream is None:
        return

    # Construye las rutas completas para las carpetas de resultados (imágenes y detecciones CSV).
    results_img_folder = os.path.join(base_output_dir_for_file, "resultados_imagenes_filtrados")
    results_folder = os.path.join(base_output_dir_for_file, "resultados_detecciones_filtrados")

    # Asegura que las carpetas de resultados existan, creándolas si es necesario.
    os.makedirs(results_img_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Define los parámetros de los filtros de banda de paso a aplicar.
    filters = [
        {"type": "0.5-2Hz", "freqmin": 0.5, "freqmax": 2},
        {"type": "2-4Hz", "freqmin": 2, "freqmax": 4},
        {"type": "5-10Hz", "freqmin": 5, "freqmax": 10},
        {"type": "1-15Hz", "freqmin": 1, "freqmax": 15}
    ]

    # Extrae el nombre base del archivo (sin ruta ni extensión).
    basename = os.path.splitext(os.path.basename(filepath))[0]

    # --- Procesamiento de la señal original ---
    print("Procesando señal original...")
    predictions_dict = {
        "original": process_stream_with_models(
            original_stream, pn_model, eqt_model, gpd_model,
            basename, results_folder, "original" # Tipo de filtro "original"
        )
    }

    # --- Procesamiento de cada señal filtrada ---
    # `filtered_streams` almacenará los streams filtrados para la graficación comparativa.
    filtered_streams = {}

    for filter_params in filters:
        print(f"Procesando señal con filtro {filter_params['type']}...")

        # Aplica el filtro al stream original (se crea una copia internamente en `apply_filter`).
        filtered_stream = apply_filter(original_stream, filter_params)
        if not filtered_stream: # Si el filtro falla o no retorna stream, pasa al siguiente.
            continue

        # Procesa el stream filtrado con los modelos y guarda las predicciones.
        predictions_dict[filter_params['type']] = process_stream_with_models(
            filtered_stream, pn_model, eqt_model, gpd_model,
            basename, results_folder, filter_params['type']
        )

        # Guarda el stream filtrado en el diccionario para su uso posterior en la graficación.
        filtered_streams[filter_params['type']] = filtered_stream

        # Forzar la recolección de basura después de procesar cada stream filtrado
        # para liberar memoria lo antes posible.
        gc.collect()

    # --- Generación de gráficos ---
    
    # Crea la carpeta para los gráficos comparativos.
    comparison_folder = os.path.join(results_img_folder, "comparison")
    os.makedirs(comparison_folder, exist_ok=True)

    # Genera gráficos individuales para cada tipo de filtro en todas las ventanas.
    generate_individual_plots(original_stream, filtered_streams, predictions_dict,
                             basename, results_img_folder, window_length_minutes)

    # Limpieza intermedia de memoria antes de la generación de gráficos comparativos.
    gc.collect()

    # Define la duración de la ventana y los tiempos de inicio/fin para los gráficos comparativos.
    wlength = window_length_minutes * 60
    starttime = original_stream[0].stats.starttime
    endtime = original_stream[0].stats.endtime
    total_seconds = int(endtime - starttime)

    # Genera gráficos comparativos para cada ventana de tiempo.
    window_index = 0
    for s in range(0, total_seconds, wlength):
        t0 = starttime + s
        t1 = t0 + wlength

        # Llama a la función para generar el gráfico comparativo de la ventana actual.
        plot_filtered_streams_window(
            original_stream, filtered_streams, predictions_dict,
            t0, t1, basename, window_index, comparison_folder
        )

        # Limpieza de memoria después de cada gráfico comparativo de ventana.
        gc.collect()

        window_index += 1

    # --- Limpieza final agresiva de memoria ---
    # Elimina explícitamente las referencias a objetos grandes para asegurar que se libere la memoria.
    del original_stream
    for stream_key in list(filtered_streams.keys()):
        del filtered_streams[stream_key]
    del filtered_streams

    for pred_key in list(predictions_dict.keys()):
        pred_data = predictions_dict[pred_key]
        for model_key in list(pred_data.keys()):
            del pred_data[model_key] # Elimina las predicciones de cada modelo
        del pred_data # Elimina el diccionario de predicciones por filtro
        del predictions_dict[pred_key] # Elimina la entrada del diccionario principal
    del predictions_dict

    # Fuerza una última recolección de basura.
    gc.collect()

    # Cierra todas las figuras de Matplotlib para liberar recursos gráficos.
    plt.close('all')

    print(f"Procesamiento de {basename} completado y memoria liberada")


def create_summary_csv(mseed_files, results_base_dir):
    """
    Crea un archivo CSV con un resumen de todos los resultados,
    mostrando el número de picks por modelo y tipo de filtro.
    Ahora busca los CSV dentro de las subcarpetas de cada archivo.
    """
    summary_file = os.path.join(results_base_dir, "summary_results.csv")

    with open(summary_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Encabezados
        csv_writer.writerow([
            "filename", "filter_type",
            "PhaseNet_P_picks", "PhaseNet_S_picks",
            "EQT_P_picks", "EQT_S_picks",
            "GPD_P_picks", "GPD_S_picks",
            "EQT_detections"
        ])

        # Para cada archivo procesado
        for filepath in mseed_files:
            basename = os.path.splitext(os.path.basename(filepath))[0]
            # La ruta de los resultados para este archivo específico
            file_specific_results_folder = os.path.join(results_base_dir, basename, "resultados_detecciones_filtrados")

            if not os.path.exists(file_specific_results_folder):
                print(f"Advertencia: No se encontró la carpeta de resultados para {basename} en {file_specific_results_folder}")
                continue

            # Buscar todos los CSV de resultados para este archivo dentro de su subcarpeta
            result_files = glob.glob(os.path.join(file_specific_results_folder, f"{basename}_*_*_picks.csv"))

            # Extraer tipos de filtro únicos
            filter_types = set()
            for result_file in result_files:
                parts = os.path.basename(result_file).split('_')
                if len(parts) >= 3:
                    # El tipo de filtro es el segundo elemento, e.g., "original", "0.5-2Hz"
                    filter_types.add(parts[1])
            filter_types = sorted(list(filter_types)) # Ordenar para consistencia

            # Para cada tipo de filtro
            for filter_type in filter_types:
                pn_p_picks = pn_s_picks = eqt_p_picks = eqt_s_picks = gpd_p_picks = gpd_s_picks = eqt_detections = 0

                # Contar picks PhaseNet
                pn_file = os.path.join(file_specific_results_folder, f"{basename}_{filter_type}_PhaseNet_picks.csv")
                if os.path.exists(pn_file):
                    with open(pn_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['phase'] == 'P':
                                pn_p_picks += 1
                            elif row['phase'] == 'S':
                                pn_s_picks += 1

                # Contar picks EQTransformer
                eqt_file = os.path.join(file_specific_results_folder, f"{basename}_{filter_type}_EQTransformer_picks.csv")
                if os.path.exists(eqt_file):
                    with open(eqt_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['phase'] == 'P':
                                eqt_p_picks += 1
                            elif row['phase'] == 'S':
                                eqt_s_picks += 1

                # Contar picks GPD
                gpd_file = os.path.join(file_specific_results_folder, f"{basename}_{filter_type}_GPD_picks.csv")
                if os.path.exists(gpd_file):
                    with open(gpd_file, 'r') as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row['phase'] == 'P':
                                gpd_p_picks += 1
                            elif row['phase'] == 'S':
                                gpd_s_picks += 1

                # Contar detecciones EQTransformer
                det_file = os.path.join(file_specific_results_folder, f"{basename}_{filter_type}_EQTransformer_detections.csv")
                if os.path.exists(det_file):
                    with open(det_file, 'r') as f:
                        reader = csv.DictReader(f)
                        eqt_detections = sum(1 for _ in reader) # Cuenta el número de filas (detecciones)

                # Escribir fila de resumen
                csv_writer.writerow([
                    basename, filter_type,
                    pn_p_picks, pn_s_picks,
                    eqt_p_picks, eqt_s_picks,
                    gpd_p_picks, gpd_s_picks,
                    eqt_detections
                ])

    print(f"Resumen de resultados guardado en: {summary_file}")

class SeismicProcessor:
    """
    Clase para encapsular y gestionar el flujo de procesamiento sísmico utilizando
    modelos de detección de fases de SeisBench. Proporciona una interfaz modular
    para cargar modelos, procesar múltiples archivos MiniSEED, guardar resultados
    y generar visualizaciones, con un enfoque en la eficiencia de memoria.
    """

    def __init__(self, dataset="stead"):
        """
        Inicializa la clase SeismicProcessor.

        Args:
            dataset (str, optional):
                El nombre del dataset pre-entrenado a utilizar para cargar los modelos
                de SeisBench (ej., "stead", "instance", "ethz"). Por defecto es "stead".
        """
        self.pn_model = None  # Modelo PhaseNet
        self.eqt_model = None # Modelo EQTransformer
        self.gpd_model = None # Modelo GPD
        self.models_loaded = False # Bandera para indicar si los modelos están cargados
        self.dataset = dataset # Almacena el nombre del dataset solicitado
        self.current_loaded_dataset = None # Rastrea qué dataset se cargó actualmente para evitar recargas innecesarias

    def load_models(self):
        """
        Carga los modelos de detección de fases P y S (PhaseNet, EQTransformer, GPD)
        pre-entrenados del dataset especificado en la inicialización de la clase.
        Esta función solo recargará los modelos si el dataset solicitado es diferente
        al ya cargado, o si los modelos no han sido cargados previamente.
        También intenta mover los modelos a la GPU si está disponible para acelerar la inferencia.

        Returns:
            bool: True si los modelos se cargaron (o ya estaban cargados) exitosamente,
                  False en caso de error.
        """
        # Verifica si los modelos ya están cargados para el dataset actual
        if self.models_loaded and self.pn_model and self.eqt_model and self.gpd_model and \
           self.current_loaded_dataset == self.dataset:
            print(f"Modelos para el dataset '{self.dataset}' ya cargados. Omitiendo recarga.")
            return True

        try:
            print(f"Cargando modelos pre-entrenados con el dataset: {self.dataset}...")
            # Carga cada modelo desde SeisBench
            self.pn_model = sbm.PhaseNet.from_pretrained(self.dataset)
            self.eqt_model = sbm.EQTransformer.from_pretrained(self.dataset)
            self.gpd_model = sbm.GPD.from_pretrained(self.dataset)

            # Intenta mover los modelos a la GPU para acelerar el procesamiento.
            try:
                print("Intentando mover modelos a GPU...")
                self.pn_model.cuda()
                self.eqt_model.cuda()
                self.gpd_model.cuda()
                print("Modelos cargados en GPU.")
            except Exception as e:
                # Si la GPU no está disponible o hay un error, se ejecuta en CPU.
                print(f"No se pudo mover los modelos a GPU: {e}")
                print("Ejecutando modelos en CPU...")

            self.models_loaded = True  # Marca los modelos como cargados
            self.current_loaded_dataset = self.dataset # Actualiza el dataset que fue cargado
            print(f"Modelos para el dataset '{self.dataset}' cargados exitosamente.")
            return True

        except Exception as e:
            print(f"Error cargando modelos con dataset '{self.dataset}': {e}")
            self.models_loaded = False # Asegura que el estado sea False en caso de error
            # Libera referencias a modelos incompletos o fallidos para evitar fugas de memoria
            self.pn_model = None
            self.eqt_model = None
            self.gpd_model = None
            self.current_loaded_dataset = None
            return False

    def process_files(self, mseed_files, output_base_dir, window_length_minutes=2, progress_callback=None):
        """
        Procesa una lista de archivos MiniSEED (`.mseed`) de manera secuencial.
        Para cada archivo, crea una subcarpeta dentro de `output_base_dir` para
        almacenar sus resultados específicos (CSV de picks/detecciones, imágenes
        individuales y comparativas). Esta función delega el procesamiento de
        cada archivo individual a `process_single_file`.

        Args:
            mseed_files (list):
                Una lista de cadenas, donde cada cadena es la ruta completa a un archivo MiniSEED.
            output_base_dir (str):
                El directorio raíz donde se crearán las subcarpetas para los resultados.
                Por ejemplo, si `output_base_dir` es "output_run_X", para cada archivo
                "file.mseed" se creará "output_run_X/file/" para sus resultados.
            window_length_minutes (int, optional):
                La duración de la ventana de tiempo en minutos que se utilizará para
                la generación de gráficos. Por defecto es 2 minutos.

        Returns:
            dict: Un diccionario que resume la información del procesamiento:
                  - 'total_files': Número total de archivos intentados procesar.
                  - 'processed_files': Número de archivos que se procesaron con éxito.
                  - 'base_output_directory': La ruta del directorio raíz donde se guardaron los resultados.
                  - 'summary_file': La ruta al archivo CSV que resume los resultados de todos los archivos procesados.

        Raises:
            Exception: Si los modelos de IA no han sido cargados previamente (`self.models_loaded` es False).
        """
        # Verifica que los modelos estén cargados antes de proceder.
        if not self.models_loaded:
            raise Exception(f"Los modelos no están cargados en SeismicProcessor para el dataset {self.dataset}. Por favor, llama a load_models() primero.")

        total_files = len(mseed_files)
        processed_files = [] # Lista para rastrear los archivos procesados exitosamente.

        # Itera sobre cada archivo en la lista.
        for i, filepath in enumerate(mseed_files):
            # Extrae el nombre base del archivo (sin ruta ni extensión).
            basename = os.path.splitext(os.path.basename(filepath))[0]
            # Crea una subcarpeta específica para los resultados de este archivo dentro del directorio base.
            file_output_dir = os.path.join(output_base_dir, basename)
            os.makedirs(file_output_dir, exist_ok=True)

            try:
                # Llama al callback de progreso si está definido.
                if progress_callback:
                    progress_callback(i, total_files, f"Procesando {os.path.basename(filepath)}")

                # Delega el procesamiento del archivo individual a `process_single_file`.
                self.process_single_file(
                    filepath,
                    file_output_dir, # Pasa la ruta de salida específica para este archivo.
                    window_length_minutes=window_length_minutes
                )

                processed_files.append(filepath) # Añade el archivo a la lista de procesados.

            except Exception as e:
                print(f"Error procesando {filepath}: {e}")
                # Continúa con el siguiente archivo si ocurre un error en uno.
                continue

        # Genera un archivo CSV de resumen que consolida la información de todos los archivos procesados.
        if processed_files:
            # `create_summary_csv` necesita el directorio base general para buscar los archivos de picks.
            create_summary_csv(processed_files, output_base_dir)

        # Llama al callback de progreso para indicar que el procesamiento ha finalizado.
        if progress_callback:
            progress_callback(total_files, total_files, "Procesamiento completado")

        # Forzar la recolección de basura después de que todo el batch se procesa para liberar memoria.
        gc.collect()

        return {
            'total_files': total_files,
            'processed_files': len(processed_files),
            'base_output_directory': output_base_dir, # El directorio raíz de los resultados.
            'summary_file': os.path.join(output_base_dir, "summary_results.csv") # Ruta al archivo resumen.
        }

    def process_single_file(self, filepath, base_output_dir_for_file, window_length_minutes=2):
        """
        Esta es una función auxiliar que envuelve la función global `process_file`.
        Su propósito principal es pasar los modelos de IA cargados por la clase
        (`self.pn_model`, `self.eqt_model`, `self.gpd_model`) junto con la ruta
        específica de salida para el archivo actual a la función de procesamiento.

        Args:
            filepath (str):
                La ruta completa al archivo MiniSEED (`.mseed`) a procesar.
            base_output_dir_for_file (str):
                El directorio específico donde se guardarán los resultados (CSV e imágenes)
                de este archivo individual.
            window_length_minutes (int, optional):
                La duración de la ventana en minutos para la generación de gráficos.
                Por defecto es 2 minutos.
        """
        # Llama a la función global `process_file` con todos los parámetros necesarios.
        process_file(
            filepath,
            self.pn_model,      # Modelo PhaseNet cargado por la clase
            self.eqt_model,     # Modelo EQTransformer cargado por la clase
            self.gpd_model,     # Modelo GPD cargado por la clase
            base_output_dir_for_file, # Directorio de salida específico para este archivo
            window_length_minutes
        )

    def get_image_paths(self, base_output_dir_for_file, basename):
        """
        Obtiene las rutas de todas las imágenes de gráficos generadas para un archivo
        MiniSEED específico, organizadas por tipo de visualización (original, filtrada, comparación).
        Esta función navega la estructura de directorios creada por `process_file`
        para recolectar las rutas de las imágenes.

        Args:
            base_output_dir_for_file (str):
                El directorio base que contiene los resultados de este archivo específico.
            basename (str):
                El nombre base del archivo MiniSEED (sin ruta ni extensión), utilizado para
                identificar las imágenes generadas.

        Returns:
            dict: Un diccionario donde las claves son los tipos de imágenes
                  ('original', 'filtered', 'comparison') y los valores son:
                  - 'original': Una lista de rutas a las imágenes de la señal original.
                  - 'filtered': Un diccionario donde las claves son los tipos de filtro
                                (ej., '0.5-2Hz') y los valores son listas de rutas a las
                                imágenes filtradas.
                  - 'comparison': Una lista de rutas a las imágenes de gráficos comparativos.
                  Todas las listas de rutas están ordenadas alfabéticamente.
        """
        image_paths = {
            'original': [],
            'filtered': {},
            'comparison': []
        }

        # La carpeta principal de imágenes está anidada dentro del directorio de salida del archivo.
        results_img_folder = os.path.join(base_output_dir_for_file, "resultados_imagenes_filtrados")

        # --- Buscar imágenes de la señal original ---
        original_folder = os.path.join(results_img_folder, 'original')
        if os.path.exists(original_folder):
            # Busca archivos que sigan el patrón: basename_original_window*.png
            original_files = glob.glob(os.path.join(original_folder, f"{basename}_original_window*.png"))
            image_paths['original'] = sorted(original_files) # Ordena las rutas

        # --- Buscar imágenes de señales filtradas ---
        # Define los tipos de filtro esperados para buscar sus carpetas.
        filter_types = ['0.5-2Hz', '2-4Hz', '5-10Hz', '1-15Hz']
        for filter_type in filter_types:
            filter_folder = os.path.join(results_img_folder, filter_type)
            if os.path.exists(filter_folder):
                # Busca archivos que sigan el patrón: basename_filtertype_window*.png
                filter_files = glob.glob(os.path.join(filter_folder, f"{basename}_{filter_type}_window*.png"))
                image_paths['filtered'][filter_type] = sorted(filter_files) # Almacena por tipo de filtro

        # --- Buscar imágenes de comparación ---
        comparison_folder = os.path.join(results_img_folder, 'comparison')
        if os.path.exists(comparison_folder):
            # Busca archivos que sigan el patrón: basename_comparison_window*.png
            comparison_files = glob.glob(os.path.join(comparison_folder, f"{basename}_comparison_window*.png"))
            image_paths['comparison'] = sorted(comparison_files) # Ordena las rutas

        return image_paths

# Mantener compatibilidad con el script original
def main():
    """
    Función principal (entry point) para el script de procesamiento sísmico.
    Esta función demuestra cómo utilizar la clase `SeismicProcessor` para
    automatizar el flujo de trabajo de análisis de archivos MiniSEED.
    Está diseñada para ser compatible con la ejecución directa del script.

    El flujo de trabajo incluye:
    1. Inicializar el procesador sísmico con un dataset de modelos predeterminado.
    2. Determinar las rutas de entrada y salida basadas en la ubicación del script.
    3. Crear la estructura de directorios necesaria.
    4. Buscar archivos MiniSEED en una carpeta de datos de prueba predefinida.
    5. Procesar los archivos encontrados, generando detecciones de fases y gráficos.
    6. Manejar posibles errores durante el procesamiento.

    Args:
        None: Esta función no acepta argumentos directamente.

    Returns:
        None: La función no retorna ningún valor, pero imprime mensajes de progreso
              y resultados en la consola, y guarda archivos de salida en el sistema de archivos.
    """
    # Inicializa una instancia de SeismicProcessor con el dataset "stead" por defecto.
    # Esta instancia será responsable de cargar los modelos de IA y gestionar el procesamiento.
    processor = SeismicProcessor(dataset="stead")

    # Intenta cargar los modelos de SeisBench. Si la carga falla, el script termina.
    if not processor.load_models():
        print("No se pudieron cargar los modelos. Saliendo.")
        return

    # Obtiene el directorio donde se encuentra el script actual.
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Configura la ruta a la carpeta donde se esperan encontrar los archivos MiniSEED de prueba.
    mseed_folder = os.path.join(script_dir, "test_data")
    
    # Crea un directorio único para los resultados de esta ejecución de prueba.
    # Puedes añadir un timestamp aquí (ej., f"test_run_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    # para asegurar que cada ejecución tenga su propia carpeta de resultados.
    unique_output_dir = os.path.join(script_dir, "test_run_output_")
    os.makedirs(unique_output_dir, exist_ok=True)
    
    print(f"Ruta de búsqueda de archivos mseed: {mseed_folder}")

    # Verifica si la carpeta de datos de prueba existe. Si no, la crea y pide al usuario
    # que coloque allí los archivos MiniSEED.
    if not os.path.exists(mseed_folder):
        print(f"La carpeta {mseed_folder} no existe. Creando la carpeta...")
        os.makedirs(mseed_folder, exist_ok=True)
        print(f"Por favor coloca tus archivos mseed en: {mseed_folder}")
        return

    # Busca archivos MiniSEED dentro de la carpeta de datos de prueba,
    # soportando varias extensiones comunes.
    mseed_files = []
    for ext in ["*.mseed", "*.MSEED", "*.ms", "*.MS", "*.MiniSEED", "*.miniseed"]:
        mseed_files.extend(glob.glob(os.path.join(mseed_folder, ext)))

    # Si no se encuentran archivos MiniSEED, imprime un mensaje y termina.
    if not mseed_files:
        print(f"No se encontraron archivos mseed en la carpeta: {mseed_folder}")
        return

    # Procesa los archivos MiniSEED encontrados utilizando el método `process_files`
    # del `SeismicProcessor`. Se utiliza una longitud de ventana predeterminada para los gráficos.
    try:
        # Pasa `unique_output_dir` como el directorio base donde `process_files`
        # creará subcarpetas para cada archivo.
        results = processor.process_files(mseed_files, unique_output_dir, window_length_minutes=2)
        print(f"Procesamiento completado. Resumen de resultados: {results}")
    except Exception as e:
        print(f"Error fatal durante el procesamiento: {e}")
# se ejecuta directamente (no cuando se importa como un módulo en otro script).
if __name__ == '__main__':
    main()