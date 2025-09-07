// Funciones adicionales para la aplicación sísmica

class NotificationManager {
    static show(message, type = 'info', duration = 5000) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideIn 0.3s ease reverse';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, duration);
    }
}

class UploadManager {
    constructor() {
        this.maxFileSize = 500 * 1024 * 1024; // 500MB
        this.allowedExtensions = ['mseed', 'MSEED', 'ms', 'MS', 'miniseed', 'MiniSEED'];
    }
    
    validateFiles(files) {
        const errors = [];
        let totalSize = 0;
        
        for (let file of files) {
            // Verificar extensión
            const extension = file.name.split('.').pop();
            if (!this.allowedExtensions.includes(extension)) {
                errors.push(`${file.name}: Extensión no válida`);
                continue;
            }
            
            // Verificar tamaño individual
            if (file.size > this.maxFileSize) {
                errors.push(`${file.name}: Archivo demasiado grande`);
                continue;
            }
            
            totalSize += file.size;
        }
        
        // Verificar tamaño total
        if (totalSize > this.maxFileSize) {
            errors.push('El tamaño total de los archivos excede 500MB');
        }
        
        return {
            valid: errors.length === 0,
            errors: errors,
            totalSize: totalSize
        };
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

class ProgressManager {
    constructor(progressBar, progressMessage) {
        this.progressBar = progressBar;
        this.progressMessage = progressMessage;
        this.interval = null;
    }
    
    start(jobId, callback) {
        this.interval = setInterval(async () => {
            try {
                const response = await fetch(`/progress/${jobId}`);
                const data = await response.json();
                
                if (response.ok) {
                    this.update(data);
                    
                    if (data.completed) {
                        this.stop();
                        callback(data);
                    }
                } else {
                    this.stop();
                    callback({ error: true, message: data.error || 'Error desconocido' });
                }
            } catch (error) {
                this.stop();
                callback({ error: true, message: 'Error de conexión' });
            }
        }, 1500);
    }
    
    update(data) {
        const percentage = Math.min(data.percentage || 0, 100);
        this.progressBar.style.width = percentage + '%';
        this.progressBar.textContent = percentage + '%';
        this.progressMessage.textContent = data.message || 'Procesando...';
        
        // Cambiar color según el progreso
        if (percentage < 30) {
            this.progressBar.style.background = 'linear-gradient(45deg, #dc3545, #fd7e14)';
        } else if (percentage < 70) {
            this.progressBar.style.background = 'linear-gradient(45deg, #fd7e14, #ffc107)';
        } else {
            this.progressBar.style.background = 'linear-gradient(45deg, #28a745, #20c997)';
        }
    }
    
    stop() {
        if (this.interval) {
            clearInterval(this.interval);
            this.interval = null;
        }
    }
}

// Utilidades para manejo de errores
class ErrorHandler {
    static handle(error, context = '') {
        console.error(`Error en ${context}:`, error);
        
        let message = 'Ha ocurrido un error inesperado';
        
        if (error.message) {
            message = error.message;
        } else if (typeof error === 'string') {
            message = error;
        }
        
        NotificationManager.show(message, 'error');
    }
    
    static handleNetworkError() {
        NotificationManager.show(
            'Error de conexión. Por favor verifica tu conexión a internet.',
            'error'
        );
    }
    
    static handleFileError(filename, error) {
        NotificationManager.show(
            `Error procesando ${filename}: ${error}`,
            'error'
        );
    }
}

// Utilidades para la interfaz
class UIUtils {
    static showElement(element, animationClass = 'fadeIn') {
        element.style.display = 'block';
        element.style.animation = `${animationClass} 0.3s ease`;
    }
    
    static hideElement(element, animationClass = 'fadeOut') {
        element.style.animation = `${animationClass} 0.3s ease`;
        setTimeout(() => {
            element.style.display = 'none';
        }, 300);
    }
    
    static setButtonLoading(button, loading = true) {
        if (loading) {
            button.disabled = true;
            button.innerHTML = `<span class="spinner me-2"></span>Procesando...`;
        } else {
            button.disabled = false;
            button.innerHTML = button.getAttribute('data-original-text') || 'Procesar';
        }
    }
    
    static addSpinner(element) {
        const spinner = document.createElement('span');
        spinner.className = 'spinner me-2';
        element.prepend(spinner);
    }
    
    static removeSpinner(element) {
        const spinner = element.querySelector('.spinner');
        if (spinner) {
            spinner.remove();
        }
    }
}

// Funciones de animación
class AnimationUtils {
    static typeWriter(element, text, speed = 50) {
        element.textContent = '';
        let i = 0;
        
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        
        type();
    }
    
    static countUp(element, target, duration = 2000) {
        const start = 0;
        const increment = target / (duration / 16);
        let current = start;
        
        const timer = setInterval(() => {
            current += increment;
            element.textContent = Math.floor(current);
            
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            }
        }, 16);
    }
    
    static pulse(element, duration = 1000) {
        element.style.animation = `pulse ${duration}ms ease-in-out`;
        setTimeout(() => {
            element.style.animation = '';
        }, duration);
    }
}

// Funciones de almacenamiento local (usando variables en memoria)
class MemoryStorage {
    constructor() {
        this.data = {};
    }
    
    set(key, value) {
        this.data[key] = JSON.stringify(value);
    }
    
    get(key) {
        const value = this.data[key];
        return value ? JSON.parse(value) : null;
    }
    
    remove(key) {
        delete this.data[key];
    }
    
    clear() {
        this.data = {};
    }
}

// Instancia global de almacenamiento en memoria
const memoryStorage = new MemoryStorage();

class DownloadManager {
    static async downloadFile(url, filename) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Error en la descarga');
            
            const blob = await response.blob();
            const downloadUrl = window.URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            window.URL.revokeObjectURL(downloadUrl);
            document.body.removeChild(a);
            
            NotificationManager.show('Descarga completada', 'success');
        } catch (error) {
            ErrorHandler.handle(error, 'descarga');
        }
    }
    
    static async downloadWithProgress(url, filename, progressCallback) {
        try {
            const response = await fetch(url);
            if (!response.ok) throw new Error('Error en la descarga');
            
            const contentLength = response.headers.get('content-length');
            const total = parseInt(contentLength, 10);
            let loaded = 0;
            
            const reader = response.body.getReader();
            const chunks = [];
            
            while (true) {
                const { done, value } = await reader.read();
                
                if (done) break;
                
                chunks.push(value);
                loaded += value.length;
                
                if (progressCallback && total) {
                    progressCallback((loaded / total) * 100);
                }
            }
            
            const blob = new Blob(chunks);
            const downloadUrl = window.URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = downloadUrl;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            
            window.URL.revokeObjectURL(downloadUrl);
            document.body.removeChild(a);
            
        } catch (error) {
            ErrorHandler.handle(error, 'descarga con progreso');
        }
    }
}

// Inicialización cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', function() {
    // Aplicar animaciones de entrada
    const animatedElements = document.querySelectorAll('.hero-section, .model-info, .filter-info, .card');
    animatedElements.forEach((el, index) => {
        setTimeout(() => {
            el.style.animation = 'fadeIn 0.6s ease forwards';
        }, index * 100);
    });
    
    // Mejorar accesibilidad con teclado
    document.addEventListener('keydown', function(e) {
        // ESC para cerrar modales o limpiar
        if (e.key === 'Escape') {
            const clearBtn = document.getElementById('clearBtn');
            if (clearBtn && !clearBtn.disabled) {
                clearBtn.click();
            }
        }
        
        // Enter para procesar cuando el botón esté enfocado
        if (e.key === 'Enter' && e.target.id === 'processBtn') {
            e.target.click();
        }
    });
    
    // Mostrar notificación de bienvenida
    setTimeout(() => {
        NotificationManager.show(
            'Bienvenido al Detector Sísmico. Sube tus archivos MSEED para comenzar.',
            'info',
            3000
        );
    }, 1000);
});