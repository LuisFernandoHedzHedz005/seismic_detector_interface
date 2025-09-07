import obspy
from obspy import read, Stream

# Método 1: Leer los tres archivos por separado y combinarlos
def merge_components_method1(file1, file2, file3, output_file):
    """
    Une tres archivos mseed de diferentes componentes en uno solo
    
    Args:
        file1, file2, file3: rutas de los archivos mseed de cada componente
        output_file: nombre del archivo de salida
    """
    # Leer cada archivo
    st1 = read(file1)
    st2 = read(file2)
    st3 = read(file3)
    
    # Crear un Stream vacío y agregar todas las trazas
    merged_stream = Stream()
    merged_stream += st1
    merged_stream += st2
    merged_stream += st3
    
    # Escribir el archivo combinado
    merged_stream.write(output_file, format='MSEED')
    
    print(f"Archivo combinado guardado como: {output_file}")
    print(f"Número total de trazas: {len(merged_stream)}")
    
    return merged_stream

# Método 2: Leer múltiples archivos de una vez
def merge_components_method2(file_list, output_file):
    """
    Une múltiples archivos mseed usando wildcards o lista de archivos
    
    Args:
        file_list: lista de archivos o patrón con wildcards
        output_file: nombre del archivo de salida
    """
    # Leer todos los archivos de una vez
    if isinstance(file_list, str):
        # Si es un patrón con wildcards (ej: "sismo_*.mseed")
        merged_stream = read(file_list)
    else:
        # Si es una lista de archivos
        merged_stream = Stream()
        for file in file_list:
            merged_stream += read(file)
    
    # Escribir el archivo combinado
    merged_stream.write(output_file, format='MSEED')
    
    print(f"Archivo combinado guardado como: {output_file}")
    print(f"Número total de trazas: {len(merged_stream)}")
    
    return merged_stream

# Método 3: Con verificación y ordenamiento de componentes
def merge_components_advanced(files_dict, output_file):
    """
    Une componentes con verificación y ordenamiento
    
    Args:
        files_dict: diccionario con componentes {'Z': 'archivo_z.mseed', 'N': 'archivo_n.mseed', 'E': 'archivo_e.mseed'}
        output_file: nombre del archivo de salida
    """
    merged_stream = Stream()
    
    # Leer y verificar cada componente
    for component, file_path in files_dict.items():
        try:
            st = read(file_path)
            print(f"Componente {component}: {len(st)} trazas cargadas")
            
            # Verificar que el componente coincida (opcional)
            for tr in st:
                if hasattr(tr.stats, 'channel') and tr.stats.channel[-1] != component:
                    print(f"Advertencia: Canal {tr.stats.channel} no coincide con componente {component}")
            
            merged_stream += st
            
        except Exception as e:
            print(f"Error al leer {file_path}: {e}")
    
    # Ordenar por estación y componente
    merged_stream.sort(['station', 'channel'])
    
    # Escribir el archivo combinado
    merged_stream.write(output_file, format='MSEED')
    
    print(f"Archivo combinado guardado como: {output_file}")
    print("Resumen de trazas:")
    for tr in merged_stream:
        print(f"  {tr.id}: {tr.stats.starttime} - {tr.stats.endtime}")
    
    return merged_stream

# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo 1: Archivos con nombres específicos
    file_z = "ANGA.PT.00.HHZ.2025.191.mseed"
    file_n = "ANGA.PT.00.HHN.2025.191.mseed"
    file_e = "ANGA.PT.00.HHE.2025.191.mseed"
    
    # Unir usando método 1
    merged = merge_components_method1(file_z, file_n, file_e, "sismo_completo.mseed")
    
    # Ejemplo 2: Usando lista de archivos
    files = [file_z, file_n, file_e]
    merged = merge_components_method2(files, "sismo_completo_v2.mseed")
    
    # Ejemplo 3: Usando wildcards (si los archivos siguen un patrón)
    # merged = merge_components_method2("sismo_*.mseed", "sismo_completo_v3.mseed")
    
    # Ejemplo 4: Método avanzado
    files_dict = {
        'Z': file_z,
        'N': file_n,
        'E': file_e
    }
    merged = merge_components_advanced(files_dict, "sismo_completo_advanced.mseed")
    
    # Verificar el resultado
    print("\nVerificación del archivo combinado:")
    st_verificacion = read("sismo_completo.mseed")
    print(st_verificacion)