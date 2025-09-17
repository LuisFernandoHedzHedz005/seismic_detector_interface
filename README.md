# Detector Sísmico con Deep Learning

Este proyecto implementa una interfaz web para la detección automática de ondas sísmicas P y S en archivos MSEED, utilizando modelos de Deep Learning como EQTransformer, PhaseNet y GPD. Permite la carga, procesamiento y visualización de resultados, facilitando el análisis sísmico para usuarios técnicos y no técnicos.

## Tabla de Contenidos

- [Características](#características)
- [Arquitectura](#arquitectura)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura de Carpetas](#estructura-de-carpetas)
- [Tecnologías](#tecnologías)
- [Citación](#Citación)
- [Contacto](#contacto)

## Características

- Detección automática de ondas sísmicas P y S.
- Integración de datasets disponibles
- Visualización interactiva de resultados y gráficas.
- Descarga de resultados procesados.
- Selección de dataset.
- Interfaz web local.

## Arquitectura

- **Backend:** Python (Flask), SeisBench, procesamiento de archivos MSEED.
- **Frontend:** HTML, CSS (Bootstrap), Jinja.
- **Modelos:** EQTransformer, PhaseNet, GPD.

## Instalación

> **Requisito previo:** Instala [Anaconda](https://www.anaconda.com/products/distribution) si aún no lo tienes. Anaconda permite crear entornos aislados y gestionar fácilmente las dependencias necesarias para este proyecto.

> **Nota:** Las versiones de PyTorch y los paquetes relacionados con NVIDIA (CUDA, cuDNN, etc.) pueden variar dependiendo del hardware y sistema operativo donde se ejecute el proyecto. Si tienes una GPU NVIDIA, asegúrate de instalar las versiones compatibles con tu tarjeta y tus drivers. Consulta la [guía oficial de PyTorch](https://pytorch.org/get-started/locally/) para más detalles.

1. Clona el repositorio:
    ```sh
    git clone https://github.com/LuisFernandoHedzHedz005/seismic_detector_interface.git
    cd seismic_detector_interface
    ```
2. Instala las dependencias:
    ```sh
    conda env create -f environment.yml
    conda activate seisbench-env
    ```
    Si necesitas modificar las versiones de PyTorch o los paquetes NVIDIA, edita el archivo `environment.yml` antes de crear el entorno.
3. Ejecuta la aplicación:
    ```sh
    python app.py
    ```

## Uso

1. Accede a `http://localhost:5000` en tu navegador.
2. Sube archivos MSEED usando la interfaz.
3. Selecciona la duración de ventana y el dataset de preentrenamiento.
4. Procesa los archivos y visualiza/descarga los resultados.

## Estructura de Carpetas

```
├── app.py                  # Aplicación principal Flask
├── seismic_processor.py    # Procesamiento sísmico y modelos
├── static/                 # Archivos estáticos (JS, CSS)
├── templates/              # Plantillas HTML/Jinja2
├── uploads/                # Archivos subidos por el usuario
├── results/                # Resultados generados
├── environment.yml         # Dependencias del entorno
└── README.md               # Este archivo
```

## Tecnologías

- Python 3.12+
- Flask
- SeisBench
- Bootstrap
- Jinja2
- JavaScript


## Citación

> Woollam, J., Münchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Bornstein, T., Diehl, T., Giuchi, C., Haslinger, F., Jozinović, D., Michelini, A., Saul, J., & Soto, H. (2022). SeisBench - A Toolbox for Machine Learning in Seismology. *Seismological Research Letters*. https://doi.org/10.1785/0220210324  
> Pre-print: https://arxiv.org/abs/2111.00786

## Contacto

Autor: Luis Fernando Hernández Hernández