from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os
import shutil
import zipfile
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
import threading
import time
import glob
from seismic_processor import SeismicProcessor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Crear carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Variables globales para el procesamiento
processing_status = {}
processor = SeismicProcessor(dataset="stead") 

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida"""
    allowed_extensions = {'mseed', 'MSEED', 'ms', 'MS', 'miniseed', 'MiniSEED'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {ext.lower() for ext in allowed_extensions}

def update_progress(job_id, current, total, message):
    """Actualiza el progreso de un trabajo"""
    processing_status[job_id] = {
        'current': current,
        'total': total,
        'message': message,
        'percentage': int((current / total) * 100) if total > 0 else 0,
        'completed': current >= total
    }

def process_files_async(job_id, mseed_files, output_dir, window_length_minutes, dataset):
    """Procesa archivos de manera asíncrona"""
    try:
        def progress_callback(current, total, message):
            update_progress(job_id, current, total, message)

        current_processor = SeismicProcessor(dataset=dataset)
        # Asegurarse de que los modelos se carguen con el dataset correcto
        if not current_processor.load_models():
            raise Exception(f"No se pudieron cargar los modelos con el dataset: {dataset}")

        # Inicializar progreso
        update_progress(job_id, 0, len(mseed_files), "Iniciando procesamiento...")

        # Procesar archivos con la duración de ventana y dataset especificados
        processor_results = current_processor.process_files(
            mseed_files,
            output_dir,
            window_length_minutes=window_length_minutes,
            progress_callback=progress_callback
        )

        # Asegurar que el diccionario de resultados contenga las rutas necesarias
        # 'output_dir' es la carpeta de resultados específica para este trabajo
        if 'images_folder' not in processor_results:
            processor_results['images_folder'] = output_dir
        if 'results_folder' not in processor_results:
            processor_results['results_folder'] = output_dir

        # Marcar como completado
        processing_status[job_id].update({
            'completed': True,
            'results': processor_results, # Usar los resultados aumentados
            'message': 'Procesamiento completado exitosamente',
            'window_length': window_length_minutes,
            'dataset': dataset
        })

    except Exception as e:
        processing_status[job_id] = {
            'current': 0,
            'total': 1,
            'message': f'Error: {str(e)}',
            'percentage': 0,
            'completed': True,
            'error': True
        }

def organize_images_by_type(job_id_full_name, results_folder): # Cambiado para aceptar el nombre completo de la carpeta del trabajo
    """
    Organiza las imágenes por tipo y por archivo MSEED original procesado.
    La estructura de la carpeta es: results_folder/<mseed_file_id>/resultados_imagenes_filtrados/filtro/imagen.png
    Las imágenes tienen nombres como: FILENAME_window_X_FILTER_detections.png
    """

    all_images_data = {}
    total_windows_per_file = {}

    if not os.path.exists(results_folder):
        print(f"La carpeta de resultados no existe: {results_folder}")
        return all_images_data, total_windows_per_file

    # El nombre de la carpeta de resultados del trabajo ya se pasa como job_id_full_name
    # Por ejemplo: '0fc400d3-0494-4833-9fee-a5863b90a63f_20250710_170549'

    # Mapear los tipos de filtros disponibles (carpetas en disco)
    filter_names_expected = [
        '0.5-2Hz', '1-15Hz', '2-4Hz', '5-10Hz', 'comparison', 'original'
    ]

    # Iterar a través de cada directorio de archivo MSEED original dentro de results_folder
    mseed_file_dirs = [d for d in os.listdir(results_folder) if os.path.isdir(os.path.join(results_folder, d))]
    
    for mseed_file_id in mseed_file_dirs:
        mseed_results_base_path = os.path.join(results_folder, mseed_file_id, 'resultados_imagenes_filtrados')

        if not os.path.exists(mseed_results_base_path):
            print(f"Debug: No se encontró la carpeta de imágenes filtradas para {mseed_file_id}: {mseed_results_base_path}")
            continue

        temp_file_grouping_for_id = {
            'comparison': [],
            'original': [],
            'filtered': {
                '0.5-2Hz': [],
                '2-4Hz': [],
                '5-10Hz': [],
                '1-15Hz': []
            }
        }

        for filter_name in filter_names_expected:
            filter_dir_path = os.path.join(mseed_results_base_path, filter_name)
            if os.path.exists(filter_dir_path):
                images_in_filter_dir = sorted(glob.glob(os.path.join(filter_dir_path, '*.png')))

                for img_full_path in images_in_filter_dir:
                    img_filename = os.path.basename(img_full_path)
                    
                    full_relative_path_for_template = os.path.join(
                        job_id_full_name,
                        mseed_file_id,
                        'resultados_imagenes_filtrados',
                        filter_name,
                        img_filename
                    )

                    if filter_name in ['comparison', 'original']:
                        temp_file_grouping_for_id[filter_name].append(full_relative_path_for_template)
                    else:
                        temp_file_grouping_for_id['filtered'][filter_name].append(full_relative_path_for_template)
        
        temp_file_grouping_for_id['comparison'].sort()
        temp_file_grouping_for_id['original'].sort()
        for f_type in temp_file_grouping_for_id['filtered']:
            temp_file_grouping_for_id['filtered'][f_type].sort()

        all_images_data[mseed_file_id] = temp_file_grouping_for_id

        windows_count = 0
        for filt_imgs in temp_file_grouping_for_id['filtered'].values():
            windows_count = max(windows_count, len(filt_imgs))
        windows_count = max(windows_count, len(temp_file_grouping_for_id['comparison']))
        windows_count = max(windows_count, len(temp_file_grouping_for_id['original']))
        
        total_windows_per_file[mseed_file_id] = windows_count

    print(f"Debug - Estructura corregida:")
    print(f"Debug - all_images_data keys: {list(all_images_data.keys())}")
    for key, data in all_images_data.items():
        print(f"Debug - {key}: comparison={len(data['comparison'])}, original={len(data['original'])}")
        for filt, imgs in data['filtered'].items():
            print(f"Debug - {key} {filt}: {len(imgs)} imágenes")

    return all_images_data, total_windows_per_file

    print(f"Debug - Estructura corregida:")
    print(f"Debug - all_images_data keys: {list(all_images_data.keys())}")
    for key, data in all_images_data.items():
        print(f"Debug - {key}: comparison={len(data['comparison'])}, original={len(data['original'])}")
        for filt, imgs in data['filtered'].items():
            print(f"Debug - {key} {filt}: {len(imgs)} imágenes")

    return all_images_data, total_windows_per_file

def count_total_windows(images_data):
    """Cuenta el total de ventanas basado en las imágenes de comparación"""
    if images_data['comparison']:
        return len(images_data['comparison'])
    elif images_data['original']:
        return len(images_data['original'])
    else:
        # Si no hay imágenes de comparación u originales, usar el primer filtro disponible
        for filter_type, filter_images in images_data['filtered'].items():
            if filter_images:
                return len(filter_images)
    return 0

def validate_window_length(window_length):
    """Valida que la duración de la ventana esté dentro de los límites permitidos"""
    try:
        minutes = int(window_length)
        if minutes < 1:
            return False, "La duración de la ventana debe ser de al menos 2 minutos"
        elif minutes > 1460:  # 24 horas = 1440 minutos + 20 minutos de margen
            return False, "La duración de la ventana no puede exceder 24 horas (1440 minutos)"
        return True, minutes
    except (ValueError, TypeError):
        return False, "La duración de la ventana debe ser un número entero"

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.j2')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Maneja la subida de archivos"""
    if 'files' not in request.files:
        return jsonify({'error': 'No se enviaron archivos'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No se seleccionaron archivos'}), 400
    
    # Obtener y validar la duración de la ventana
    window_length_raw = request.form.get('window_length', 2)
    is_valid, window_length_or_error = validate_window_length(window_length_raw)
    
    if not is_valid:
        return jsonify({'error': window_length_or_error}), 400
    
    window_length_minutes = window_length_or_error

    # Obtener el dataset seleccionado
    dataset = request.form.get('dataset', 'stead') # 'stead' como valor por defecto
    
    # Validar archivos
    valid_files = []
    for file in files:
        if file and allowed_file(file.filename):
            valid_files.append(file)
    
    if not valid_files:
        return jsonify({'error': 'No se encontraron archivos MSEED válidos'}), 400
    
    # Crear ID único para este trabajo
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Crear carpetas para este trabajo
    job_upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
    job_results_dir = os.path.join(app.config['RESULTS_FOLDER'], f"{job_id}_{timestamp}")
    
    os.makedirs(job_upload_dir, exist_ok=True)
    os.makedirs(job_results_dir, exist_ok=True)
    
    # Guardar archivos
    saved_files = []
    for file in valid_files:
        filename = secure_filename(file.filename)
        filepath = os.path.join(job_upload_dir, filename)
        file.save(filepath)
        saved_files.append(filepath)
    
    # Inicializar el estado del procesamiento con la información de la ventana y el dataset
    processing_status[job_id] = {
        'current': 0,
        'total': len(saved_files),
        'message': 'Preparando procesamiento...',
        'percentage': 0,
        'completed': False,
        'window_length': window_length_minutes,
        'dataset': dataset # Guardar el dataset en el estado del trabajo
    }
    
    # Iniciar procesamiento en hilo separado
    thread = threading.Thread(
        target=process_files_async,
        args=(job_id, saved_files, job_results_dir, window_length_minutes, dataset)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'job_id': job_id,
        'message': f'Se subieron {len(saved_files)} archivos. Procesamiento iniciado con ventanas de {window_length_minutes} minutos y dataset {dataset}.',
        'files_count': len(saved_files),
        'window_length': window_length_minutes,
        'dataset': dataset
    })

@app.route('/progress/<job_id>')
def get_progress(job_id):
    """Obtiene el progreso de un trabajo"""
    if job_id not in processing_status:
        return jsonify({'error': 'Trabajo no encontrado'}), 404
    
    return jsonify(processing_status[job_id])

@app.route('/results/<job_id>')
def show_results(job_id):
    """Muestra los resultados de un trabajo"""
    if job_id not in processing_status:
        return render_template('error.j2', error='Trabajo no encontrado'), 404
    
    status = processing_status[job_id]
    if not status.get('completed', False):
        return render_template('error.j2', error='Trabajo aún en progreso'), 400
    
    if status.get('error', False):
        return render_template('error.j2', error=status.get('message', 'Error desconocido')), 500
    
    results = status.get('results', {})
    
    # Preparar datos de resultados
    results_data = {
        'processed_files': results.get('processed_files', 0),
        'total_files': results.get('total_files', 0),
        'processing_time': results.get('processing_time', 0),
        'success': results.get('success', False),
        'window_length': status.get('window_length', 2),
        'dataset': status.get('dataset', 'stead')
    }
    
    # images_folder ya es la ruta completa como 'results/job_id_timestamp'
    images_folder = results.get('images_folder', '') 
    
    # Obtener solo el nombre de la carpeta del trabajo (job_id_timestamp)
    job_id_full_name = os.path.basename(images_folder)

    # Usar la función corregida, pasando el nombre completo de la carpeta del trabajo
    all_images_data, total_windows_per_file = organize_images_by_type(job_id_full_name, images_folder)
    
    # Debug: Imprimir datos para verificar
    print(f"Debug - images_folder: {images_folder}")
    print(f"Debug - all_images_data keys: {list(all_images_data.keys())}")
    print(f"Debug - total_windows_per_file: {total_windows_per_file}")
    
    # Calcular el total de ventanas global sumando las ventanas por archivo
    total_windows_global = sum(total_windows_per_file.values())
    
    # Si no hay ventanas, intentar contar todas las imágenes disponibles
    if total_windows_global == 0:
        print("Debug - No se encontraron ventanas, contando todas las imágenes...")
        # Contar todas las imágenes PNG en la carpeta de resultados
        all_images = []
        for root, dirs, files in os.walk(images_folder):
            for file in files:
                if file.endswith('.png'):
                    all_images.append(os.path.join(root, file))
        
        print(f"Debug - Total imágenes encontradas: {len(all_images)}")
        if all_images:
            total_windows_global = len(all_images)
            # Actualizar también el diccionario para mostrar al menos las imágenes encontradas
            if not all_images_data:
                all_images_data['detected_files'] = {
                    'comparison': [],
                    'original': [],
                    'filtered': {'0.5-2Hz': [], '2-4Hz': [], '5-10Hz': [], '1-15Hz': []}
                }
                
                # Distribuir las imágenes según su ruta
                for img_path in all_images:
                    rel_path = os.path.relpath(img_path, images_folder)
                    if 'comparison' in rel_path.lower():
                        all_images_data['detected_files']['comparison'].append(rel_path)
                    elif 'original' in rel_path.lower():
                        all_images_data['detected_files']['original'].append(rel_path)
                    elif '0.5-2hz' in rel_path.lower():
                        all_images_data['detected_files']['filtered']['0.5-2Hz'].append(rel_path)
                    elif '2-4hz' in rel_path.lower():
                        all_images_data['detected_files']['filtered']['2-4Hz'].append(rel_path)
                    elif '5-10hz' in rel_path.lower():
                        all_images_data['detected_files']['filtered']['5-10Hz'].append(rel_path)
                    elif '1-15hz' in rel_path.lower():
                        all_images_data['detected_files']['filtered']['1-15Hz'].append(rel_path)
                
                total_windows_per_file['detected_files'] = total_windows_global
    
    print(f"Debug - Total ventanas final: {total_windows_global}")
    
    return render_template('results.j2', 
                         job_id=job_id,
                         results=results_data,
                         all_images_data=all_images_data,
                         total_windows=total_windows_global,
                         total_windows_per_file=total_windows_per_file,
                         base_path=app.config['RESULTS_FOLDER'])

@app.route('/serve_image/<job_id>/<path:image_path>')
def serve_image(job_id, image_path):
    """Sirve imágenes de resultados"""
    # Construir la ruta completa de la imagen
    full_path = os.path.join(app.config['RESULTS_FOLDER'], image_path)
    
    # Verificar que el archivo existe y es seguro
    if os.path.exists(full_path) and os.path.isfile(full_path):
        return send_file(full_path)
    else:
        print(f"Debug: Imagen NO ENCONTRADA en ruta: {full_path}")
        return jsonify({'error': 'Imagen no encontrada'}), 404

@app.route('/download/<job_id>')
def download_results(job_id):
    """Descarga los resultados como ZIP"""
    if job_id not in processing_status:
        return jsonify({'error': 'Trabajo no encontrado'}), 404

    status = processing_status[job_id]
    if not status.get('completed', False) or status.get('error', False):
        return jsonify({'error': 'Resultados no disponibles'}), 400

    results = status.get('results', {})
    if 'results_folder' not in results or not os.path.exists(results['results_folder']):
        return jsonify({'error': 'Carpeta de resultados no encontrada o no válida'}), 404

    job_specific_results_dir = results['results_folder']
    window_length = status.get('window_length', 2)
    dataset_name = status.get('dataset', 'stead')
    zip_filename = f"resultados_{job_id}_ventana_{window_length}min_dataset_{dataset_name}.zip"
    zip_path = os.path.join(app.config['RESULTS_FOLDER'], zip_filename)

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(job_specific_results_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, job_specific_results_dir)
                    zipf.write(file_path, arcname)

        return send_file(zip_path, as_attachment=True, download_name=zip_filename)

    except Exception as e:
        return jsonify({'error': f'Error creando ZIP: {str(e)}'}), 500

@app.route('/image/<path:image_path>')
def serve_image_legacy(image_path):
    """Sirve imágenes de resultados (ruta legacy)"""
    full_path = os.path.join(app.config['RESULTS_FOLDER'], image_path)
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        return jsonify({'error': 'Imagen no encontrada'}), 404

@app.route('/cleanup/<job_id>', methods=['POST'])
def cleanup_job(job_id):
    """Limpia los archivos de un trabajo"""
    try:
        # Eliminar archivos subidos
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], job_id)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        
        # Eliminar de memoria
        if job_id in processing_status:
            del processing_status[job_id]
        
        return jsonify({'message': 'Trabajo limpiado exitosamente'})
    
    except Exception as e:
        return jsonify({'error': f'Error limpiando trabajo: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'Archivo demasiado grande. Máximo 500MB'}), 413

if __name__ == '__main__':
    print("Iniciando aplicación Flask...")
    print("Cargando modelos sísmicos (dataset por defecto 'stead')...")
    
    # Cargar modelos al inicio (opcional, se pueden cargar on-demand)
    try:
        processor.load_models() 
        print("Modelos cargados exitosamente")
    except Exception as e:
        print(f"Advertencia: No se pudieron cargar los modelos al inicio: {e}")
        print("Los modelos se cargarán cuando sea necesario con el dataset seleccionado por el usuario.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
