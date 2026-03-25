# Usamos la misma imagen base ligera
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Aprovechamos la caché de Docker copiando primero los requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos el resto del código del frontend
COPY . .

# Exponemos el puerto por defecto de Gradio
EXPOSE 7860

# Variable de entorno con la URL del backend (por defecto apunta a localhost)
ENV BACKEND_URL=http://localhost:8000

# Comando para arrancar la app
CMD ["python", "gradio_app.py"]