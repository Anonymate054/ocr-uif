# Lanzadores de la Aplicación OCR-UIF / OCR-UIF Launchers

Este directorio contiene los archivos necesarios para ejecutar la aplicación final de OCR-UIF en Windows, Linux y macOS.

## 1. Windows
La aplicación para Windows es un ejecutable nativo compilado completamente offline. No requiere instalación de Python ni de dependencias adicionales.
- Simplemente haz doble clic en `OCR-UIF.exe`.

## 2. Linux
Para ejecutar la aplicación en Linux, utiliza el script `run_linux.sh`. Este script iniciará la aplicación usando el entorno virtual configurado en la raíz del proyecto.
- Ejecuta en la terminal:
  ```bash
  ./run_linux.sh
  ```

## 3. macOS
Para ejecutar la aplicación en macOS, utiliza el archivo `run_mac.command`.
- Haz doble clic en `run_mac.command` o ejecútalo en la terminal:
  ```bash
  ./run_mac.command
  ```

---

## 4. Notas / Notes
- Los modelos de NLP y de OCR están completamente integrados dentro del ejecutable de Windows y cargados localmente en los scripts de Linux y macOS.
- La aplicación no realiza conexiones externas de red (100% offline y segura).
