#*************************************************************************************************************
#***************************************  MANEJO DE AUDIO  ***************************************************
#*************************************************************************************************************

def transcribir_audio(audio_path):
    """
    Esta función transcribe un archivo de audio, calcula varias características acústicas
    (como el tempo, la tasa de cruces por cero, la energía, la envolvente, etc.), y organiza 
    los resultados en un DataFrame de pandas.

    Parámetros:
    - audio_path (str): Ruta del archivo de audio que se desea procesar y transcribir.

    Retorna:
    - pd.DataFrame: Un DataFrame con los resultados de la transcripción y las características acústicas.
    """
    
    import librosa
    import librosa.display
    import numpy as np
    import whisper
    import pandas as pd
    import warnings

    # Cargar el archivo de audio con librosa
    y, sr = librosa.load(audio_path)  # 'y' es el vector de audio y 'sr' es la tasa de muestreo
    
    # Calcular la duración del audio en segundos
    # Devuelve la duración total del audio en segundos (tiempo total de la señal)
    duration = librosa.get_duration(y=y, sr=sr)

    # 1. Detección del Tempo (ritmo)
    # Calcula el tempo (ritmo) de la señal de audio en pulsos por minuto (BPM)
    # 'tempo' es el ritmo en BPM, y el segundo valor (ignorado con "_") es la confianza en la detección.
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)  
    tempo = float(tempo[0])  # Convierte el tempo a una variable float en lugar de numpy array

    # 2. Cálculo de la tasa de cruces por cero (ZCR)
    # La ZCR indica cuántas veces la señal cruza el valor cero, lo que puede ser útil para detectar sonidos percutivos
    zcr = librosa.feature.zero_crossing_rate(y)  # Calcula la tasa de cruces por cero (ZCR) para la señal de audio
    average_zcr = np.mean(zcr)  # Calcula el valor promedio de la ZCR para obtener una medida de frecuencia de cruces

    # 3. Energía del audio
    # La energía está relacionada con la amplitud de la señal, y se calcula como la suma de los cuadrados de las muestras
    energy = np.sum(y**2)  # Calcula la energía total de la señal (suma de los cuadrados de las muestras)

    # 4. Estimación de respiración rápida (Frecuencia de la respiración)
    # Para detectar respiración rápida o cambios bruscos en la señal de audio, se calcula la envolvente y la fuerza de los onset
    # Los "onsets" corresponden a transiciones o cambios bruscos en la señal que podrían ser indicativos de respiración rápida o estrés
    envelope = librosa.onset.onset_strength(y=y, sr=sr)  # Calcula la fuerza de los onset (transiciones o cambios bruscos)
    average_envelope = np.mean(envelope)  # Calcula el valor promedio de la envolvente para detectar cambios abruptos

    # Transcripción y traducción del audio usando el modelo Whisper
    # Cargar el modelo Whisper preentrenado para la transcripción automática de voz a texto
    model = whisper.load_model("base")  # Cargar el modelo Whisper, usando "base" por defecto (puedes elegir entre otros como "tiny", "small", "medium", "large")
    
    # Transcribir el audio en español (el modelo también puede detectar y traducir otros idiomas si es necesario)
    result = model.transcribe(audio_path)  # Transcribe el archivo de audio al idioma español
    transcription = result['text']  # Extrae el texto transcrito del resultado

    # Detectar el idioma original del audio (Whisper detecta automáticamente el idioma)
    language_detected = result['language']

    # Calcular el número de palabras en la transcripción
    word_count = len(transcription.split())  # Cuenta el número de palabras en la transcripción
    
    #Determinar en variable binaria si el audio indica signos de estres
    if average_envelope > 2.02 and tempo > 90 and energy > 1e3:
        estres=1
    else:
        estres=0

    # Suprimir los warnings para evitar mensajes innecesarios durante la ejecución
    warnings.filterwarnings("ignore", category=FutureWarning)  # Suprime advertencias sobre características futuras
    warnings.filterwarnings("ignore", category=UserWarning, message="FP16 is not supported on CPU")  # Suprime advertencias sobre el uso de FP16 en CPU

    # Crear un DataFrame con las características del audio y la transcripción
    # Inicializamos un DataFrame vacío con las columnas correspondientes
    df = pd.DataFrame(columns=['nombre_audio', 'texto', 'idioma_audio', 'palabras', 'duracion', 'tempo', 'zcr', 'energia', 'picos_env','estres'])
    
    # Llenamos el DataFrame con los resultados calculados
    datos = {
        'nombre_audio': audio_path,  # Ruta del archivo de audio
        'texto': transcription,  # Texto transcrito del audio
        'idioma_audio': language_detected,  # Idioma detectado del audio
        'palabras': word_count,  # Número total de palabras en la transcripción
        'duracion': duration,  # Duración total del audio en segundos
        'tempo': tempo,  # Tempo o ritmo del audio en BPM
        'zcr': average_zcr,  # Promedio de la tasa de cruces por cero
        'energia': energy,  # Energía total de la señal de audio
        'picos_env': average_envelope,  # Promedio de la envolvente del audio
        'estres':estres #estres en el audio
    }

    # Convertir el diccionario con los resultados a un DataFrame y devolverlo
    df = pd.DataFrame([datos])  # Convierte el diccionario de resultados en un único DataFrame

        # Guardar el DataFrame actualizado en el archivo JSON
    json_path = 'audios_data.json'
    df.to_json(json_path, orient='records', lines=True)
    
    # Retorna el DataFrame con todos los resultados calculados
    return json_path  # Devuelve el DataFrame con la transcripción y características acústicas


#*************************************************************************************************************
#***************************************  CLASIFICACIÓN DE UNA EMERGENCIA  ***********************************
#*************************************************************************************************************

def clasificar_emergencia(archivo_json):
    from openai import OpenAI
    import json
    import os
    from dotenv import load_dotenv
    # Cargar archivo .env con el api_key
    load_dotenv()

    # Obtener la clave API
    api_key = os.getenv("api_key_openai")
    if not api_key:
        raise ValueError("The 'api_key' environment variable is not defined in the .env file.")

    # Configura tu clave API de OpenAI aquí
    client = OpenAI(api_key=api_key)

    try:
        # Abrir el archivo JSON y cargar los datos
        with open(archivo_json, 'r') as f:
            data = json.load(f)  # Cargar el archivo JSON como un diccionario
    except json.JSONDecodeError:
        # Si el archivo JSON no es válido o está mal formado
        print("Error al leer el archivo JSON. Asegúrate de que el archivo esté correctamente formado.")
        return []
    except FileNotFoundError:
        # Si el archivo no es encontrado
        print(f"El archivo {archivo_json} no fue encontrado.")
        return []

    try:
        # Llamada a la API de OpenAI para analizar el mensaje usando el endpoint de chat
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "Eres un asistente que clasifica mensajes de emergencia para un chatbot de seguridad en minería subterránea.\n\nPor ejemplo:\n- {\"nombre_audio\":\"voice_message.wav\",\"texto\":\" En el revaje de la rampa 5 hay un incendio para donde corremos, dinos r\\u00e1pido hacia donde corremos.\",\"idioma_audio\":\"es\",\"palabras\":18,\"duracion\":5.44,\"tempo\":95.703125,\"zcr\":0.0752368684,\"energia\":1135.146484375,\"picos_env\":2.444185257,\"estres\":1} respuesta: {emergencia:1, tipo:incendio}\n\nOtro ejemplo:\n- {\"nombre_audio\":\"voice_message.wav\",\"texto\":\" Est\\u00e1 saliendo humo de un tubo. Eso es normal, ayuda.\",\"idioma_audio\":\"es\",\"palabras\":10,\"duracion\":5.6135147392,\"tempo\":135.9991776316,\"zcr\":0.0468205223,\"energia\":365.2601318359,\"picos_env\":1.7363413572,\"estres\":0} respuesta: {emergencia:1, tipo:incendio}\n\nOtro ejemplo:\n- {\"nombre_audio\":\"voice_message.wav\",\"texto\":\" Hay un compañero tirado en el rebaje, que hago.\",\"idioma_audio\":\"es\",\"palabras\":9,\"duracion\":5.6135147392,\"tempo\":135.9991776316,\"zcr\":0.0468205223,\"energia\":365.2601318359,\"picos_env\":1.7363413572,\"estres\":1} respuesta: {emergencia:1, tipo: salud general}\n\nOtro ejemplo:\n- {\"nombre_audio\":\"voice_message.wav\",\"texto\":\" Necesito encontrar un baño, es una emergencia.\",\"idioma_audio\":\"es\",\"palabras\":7,\"duracion\":5.6135147392,\"tempo\":135.9991776316,\"zcr\":0.0468205223,\"energia\":365.2601318359,\"picos_env\":1.7363413572,\"estres\":1} respuesta: {emergencia:0, tipo: no_emergencia, idioma:es}\n\nOtro ejemplo:\n- {\"nombre_audio\":\"voice_message.wav\",\"texto\":\" Las bombas centrales de agua se apagaron.\",\"idioma_audio\":\"es\",\"palabras\":7,\"duracion\":5.6135147392,\"tempo\":135.9991776316,\"zcr\":0.0468205223,\"energia\":365.2601318359,\"picos_env\":1.7363413572,\"estres\":0} respuesta: {emergencia:1, tipo: inundación, idioma:es}\n\nOtro ejemplo:\n- {\"nombre_audio\":\"voice_message.wav\",\"texto\":\"Que hago si hay fuego en el taller y sale mucho humo.\",\"idioma_audio\":\"es\",\"palabras\":7,\"duracion\":5.6135147392,\"tempo\":135.9991776316,\"zcr\":0.0468205223,\"energia\":365.2601318359,\"picos_env\":1.7363413572,\"estres\":0} respuesta: {emergencia:0, tipo: no_emergencia, idioma:es}"
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Clasifica el siguiente mensaje de voz como una emergencia o no emergencia. Si es una emergencia, identifica el tipo (por ejemplo, incendio, inundación, caída de roca, enfermedad general, etc. Además, incluye el idioma del texto) {data}"
                        }
                    ]
                }
            ],
            response_format={"type": "text"},
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        # Obtener la respuesta generada por GPT-4
        respuesta = response.choices[0].message.content  # Extraer la respuesta de la API
        return respuesta
    except Exception as e:
        print(f"Error al procesar la API de OpenAI: {e}")
        return ""



#*************************************************************************************************************
#***************************************  RESPUESTA CON CHAT GPT  ++++++++++**********************************
#*************************************************************************************************************

def cgurin_responde(query, emergencia=None, last_update=None):
    """
    Esta función responde a consultas relacionadas con la seguridad en la minería subterránea, haciendo uso de
    una base de datos vectorial (Pinecone) y un modelo de lenguaje (OpenAI). Dependiendo del tipo de consulta,
    se puede retornar información sobre planos de los diferentes niveles de la mina, información general, 
    o respuestas detalladas en caso de emergencias mineras, como incendios, fracturas o condiciones peligrosas.

    Parámetros:
    - query (str): La consulta del usuario relacionada con la seguridad minera, nivel de la mina, o emergencia.
    - emergencia (dict, opcional): Un diccionario con información adicional sobre el tipo de emergencia (por ejemplo,
      incendio, fractura, etc.), si está disponible. Si se proporciona, se incluye en el análisis de la respuesta.
    - last_update (dict, opcional): Un diccionario con el estado más reciente de los sensores de la mina (por ejemplo, 
      niveles de CO, estado de ventilación, etc.). Esta información se incorpora para mejorar la respuesta ante emergencias.

    Retorna:
    - str: Una respuesta generada que puede incluir un plano, instrucciones sobre cómo actuar en una emergencia, o información
           general sobre la mina.
    """
    
    import os
    from dotenv import load_dotenv
    from openai import OpenAI  # API de OpenAI

    # Cargar archivo .env con las claves de API
    load_dotenv()

    # Obtener la clave API de Pinecone
    api_key = os.getenv("api_key_pinecone")
    if not api_key:
        raise ValueError("La variable de entorno 'api_key' no está definida en el archivo .env.")

    # Inicializar Pinecone con la clave API
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=api_key)

    # Obtener la clave API de OpenAI
    api_key = os.getenv("api_key_openai")
    if not api_key:
        raise ValueError("La variable de entorno 'api_key' no está definida en el archivo .env.")

    # Configura la clave API de OpenAI
    client = OpenAI(api_key=api_key)
    
    # Convertir la consulta a minúsculas para normalizar la entrada
    query_lower = query.lower()

    import re

    # Definir los niveles disponibles en la mina
    niveles = ["1950", "1850", "1810", "1750", "1900", "general"]

    # Extraer el número de nivel de la consulta utilizando una expresión regular
    number_in_query = re.search(r'\d+', query)
    nivel = number_in_query.group() if number_in_query else "general"
    
    # Definir el nombre del índice en Pinecone
    index_name = "info-mina"
    index = pc.Index(index_name)
    print(query_lower)

    # Caso en que la consulta busca información sobre un plano de algún nivel
    if ('plano' in query_lower or 'lageplan' in query_lower or 'plane' in query_lower) and any(nivel in query_lower for nivel in niveles):
        # Llamada a OpenAI para obtener el embedding de la consulta
        response_imagen = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )

        # Realizar la consulta al índice vectorial de Pinecone para buscar resultados de planos
        results_imagen = index.query(
            namespace="ns1",
            vector=response_imagen.data[0].embedding,
            top_k=3,  # Número de resultados a devolver
            include_values=True,
            include_metadata=True,
            filter={"imagen_b64": {"$ne": None, "$ne": '', "$ne": " "}}  # Filtrar por aquellos que tengan imagen en base64
        )

        import base64
        import tempfile
        from PIL import Image
        from io import BytesIO

        # Filtrar los resultados para asegurarse de que coincidan con el nivel y contengan "plano"
        filtered_results = []
        if nivel:  # Verificar si 'nivel' tiene un valor válido antes de realizar el filtrado
            for match in results_imagen['matches']:
                if nivel in match['id'] and 'plano' in match['id']:
                    filtered_results.append(match)

        # Verificar si hay resultados filtrados
        if filtered_results:
            # Extraer la imagen en base64 del primer resultado filtrado
            imagen_b64 = filtered_results[0]["metadata"].get("imagen_b64", "")
            
            if imagen_b64:  # Solo proceder si se encuentra una imagen en base64 válida
                # Decodificar la imagen base64
                imagen_data = base64.b64decode(imagen_b64)
                
                # Convertir a formato de imagen con PIL
                image = Image.open(BytesIO(imagen_data))
                
                # Crear un archivo temporal y guardar la imagen
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
                    image.save(temp_file, format="PNG")  # Guardar la imagen en el archivo temporal
                    temp_file_path = temp_file.name  # Obtener la ruta del archivo temporal
                
                # Retornar la ruta del archivo temporal
                return temp_file_path

        # Si no se encuentra una imagen, devolver texto o respuesta estándar
        return "No se encontró un plano para la consulta."
    else:
        # Si la consulta no es sobre un plano, se procede a consultar la información textual
        # Llamada a OpenAI para obtener el embedding de la consulta
        response = client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )

        # Realizar la consulta al índice vectorial de Pinecone para buscar información relevante
        results = index.query(
            namespace="ns1",
            vector=response.data[0].embedding,
            top_k=5,  # Número de resultados a devolver
            include_values=False,
            include_metadata=True
        )

        # Filtrar los resultados para asegurar que el 'nivel' esté en el 'id' y que no contengan imágenes
        filtered_results = []
        for match in results['matches']:
            imagen_b64 = match['metadata'].get("imagen_b64", "")
            if imagen_b64 in [None, "", " "]:  # Solo considerar resultados sin imagen
                #if nivel and nivel in match['id']:  # Asegurarse que el nivel esté presente en el ID
                filtered_results.append(match)

        # Si hay resultados filtrados, preparar el mensaje para el modelo GPT
        if filtered_results:
            messages = [
                {
                    "role": "system",  # Mensaje del sistema con información básica sobre el contexto
                    "content": "Eres un experto en seguridad de minería subterránea en el sistema de inteligencia ante emergencias de mina 'El Cosmo'. Tienes amplio conocimiento en primeros auxilios. - Niveles: 2000 (entrada, rampa 15%), 1950 (desarrollo, 6 personas), 1900 (taller, 20 personas), 1850 (producción, refugio minero para 15 personas), 1810 (acarreo, 4 personas), 1750 (profundización, 6 personas). - Ventilación: V1 (extractor), V2 (extractor), V3 (inyector). Sensores CO: indicados como co_nivel:0.1% (ej. co_1900:0.1%). - Señalización: Verde (hacia salida al subir rampa), Amarilla (al bajar rampa y zonas de operaciones). - Cada nivel tiene acceso a la rampa y al menos un baño ubicado en el crucero hacia la rampa. Responde conciso y seguro a las preguntas, que pueden ser de emergencia o solo de consulta."
                },
                {
                    "role": "user",  # Mensaje del usuario con la consulta
                    "content": f"El usuario te dará un mensaje como este: \"{query}\". Responde según la información de emergencia que tienes disponible, que es la siguiente: {filtered_results}. Además, consulta siempre el diccionario de estados de los sensores y equipos de la mina que es: {last_update}. Si es una emergencia, manda breve mensaje de calma y explica cómo ponerse a salvo. Otro ejemplo de pregunta puede ser: cómo llego de un nivel a otro o de un punto a otro. Responde qué hacer, la señalización y la distancia. Otra pregunta: donde hay un baño. Responde muy conciso: Cada nivel de la mina tiene al menos un baño ubicado en."
                }
            ]
            
            # Si el diccionario 'emergencia' se proporciona, agregarlo al mensaje
            if emergencia:
                messages.append({
                    "role": "user",  # Mensaje adicional con detalles de la emergencia
                    "content": f"Información adicional sobre la emergencia: {emergencia}"
                })

            # Llamada a OpenAI para obtener una respuesta basada en el modelo GPT-4
            response = client.chat.completions.create(
                model="gpt-4o",  # Asegúrate de que estés utilizando el modelo adecuado
                messages=messages,
                temperature=0.9,
                max_tokens=2048,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )

            # Almacenar y retornar la respuesta generada por GPT
            print(response)
            print(last_update)
            respuesta = response.choices[0].message.content
            return respuesta

        else:
            # Si no hay resultados relevantes, retornar un mensaje de disculpa
            return "Disculpa, desconozco esa información. Pregunta de otra vez o verifica con el jefe del área."



#*************************************************************************************************************
#*************************************  RESPUESTA DE AUDIO  **************************************************
#*************************************************************************************************************

def respuesta_audio(respuesta):

    from pathlib import Path
    import os
    from dotenv import load_dotenv
    from openai import OpenAI
    import tempfile

    # Cargar archivo .env con el api_key
    load_dotenv()

    # Obtener la clave API
    api_key = os.getenv("api_key_openai")
    if not api_key:
        raise ValueError("The 'api_key' environment variable is not defined in the .env file.")

    # Configura tu clave API de OpenAI aquí
    client = OpenAI(api_key=api_key)

    # Crear un archivo temporal para almacenar la respuesta de voz generada
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio_file:
        # Generar el audio a partir de la respuesta usando OpenAI TTS (Text-to-Speech)
        response = client.audio.speech.create(
            model="tts-1",
            voice="onyx",
            input=respuesta,
        )

        # Guardar el archivo de audio generado
        response.stream_to_file(temp_audio_file.name)  # Guardar el archivo directamente en el archivo temporal
        
        temp_audio_file_path = temp_audio_file.name  # Obtener la ruta del archivo generado
    
    return temp_audio_file_path  # Retornamos la ruta del archivo de audio generado





#*************************************************************************************************************
#*********************************  FUNCIONES PARA SECUENCIA DEL CHATBOT  ************************************
#*************************************************************************************************************

import os
import requests
from pydub import AudioSegment
from io import BytesIO
import json
import numpy as np
from estados import update_and_return_data
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import CallbackContext
import asyncio

# Función para guardar el archivo de voz y convertirlo a WAV
async def save_and_convert_voice(voice_file_id: str, bot_token: str):
    url = f"https://api.telegram.org/bot{bot_token}/getFile?file_id={voice_file_id}"
    response = requests.get(url)
    file_path = response.json()['result']['file_path']
    file_url = f"https://api.telegram.org/file/bot{bot_token}/{file_path}"

    # Descargar el archivo de voz
    voice_file = requests.get(file_url)
    
    # Convertir el archivo de audio (si es necesario)
    audio = AudioSegment.from_ogg(BytesIO(voice_file.content))  # Asumimos que el archivo es OGG
    audio.export("voice_message.wav", format="wav")
    
    return "voice_message.wav"

# Función para manejar el inicio del bot y mostrar el menú inicial
async def start(update: Update, context: CallbackContext):
    keyboard = [
        [InlineKeyboardButton("Haz una pregunta", callback_data='ask_question')],
        [InlineKeyboardButton("Estado de la mina", callback_data='estado')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.message:
        await update.message.reply_text('¡Hola! Soy C-GurIN. ¿En qué puedo ayudarte?', reply_markup=reply_markup)
    elif update.callback_query:
        await update.callback_query.edit_message_text('¡Hola! Soy C-GurIN. ¿En qué puedo ayudarte?', reply_markup=reply_markup)

# Función para manejar la opción "Haz una pregunta"
async def handle_ask_question(update: Update, context: CallbackContext):
    if update.message:
        await update.message.reply_text("Por favor, envía un mensaje de voz o de texto.")
    elif update.callback_query:
        await update.callback_query.message.reply_text("Por favor, envía un mensaje de voz o de texto.")
    
    context.user_data['awaiting_voice'] = True

# Función consolidada para manejar tanto mensajes de voz como de texto ---> Función más importante
async def handle_message(update: Update, context: CallbackContext):
    if context.user_data.get('awaiting_voice', False):
        if update.message.voice:
            print("Mensaje de voz recibido.")
            voice = update.message.voice
            file_id = voice.file_id
            voice_file = await context.bot.get_file(file_id)
            file_path = voice_file.file_path
            file_name = "user_voice_message.ogg"
            await voice_file.download_to_drive(file_name)

            # Transcribir el audio
            json_path = transcribir_audio(file_name)
            print(f"Archivo de audio transcrito: {json_path}")

            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            texto = json_data.get("texto", "")
            emergencia = clasificar_emergencia(json_path)
            print(f"Emergencia clasificada: {emergencia}")

            respuesta = cgurin_responde(query=texto, emergencia=emergencia, last_update=context.user_data.get('last_update'))
            print(f"Respuesta generada: {respuesta}")

            # Verificar si la respuesta es una imagen
            if isinstance(respuesta, str) and respuesta.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with open(respuesta, 'rb') as img_file:
                        await update.message.reply_photo(photo=img_file)
                except FileNotFoundError:
                    await update.message.reply_text("Lo siento, no se pudo encontrar la imagen.")
            else:
                try:
                    audio_respuesta = respuesta_audio(respuesta)
                    print(f"Respuesta en audio generada: {audio_respuesta}")

                    with open(audio_respuesta, 'rb') as audio_file:
                        await update.message.reply_audio(audio=audio_file)
                except FileNotFoundError:
                    await update.message.reply_text("Lo siento, no se pudo generar respuesta de audio.")
                    await update.message.reply_text(f"Respuesta de C-GurIN:\n{respuesta}")
            
            context.user_data['awaiting_voice'] = False

        elif update.message.text:
            print(f"Mensaje de texto recibido: {update.message.text}")
            query = update.message.text

            respuesta = cgurin_responde(query=query, emergencia=None, last_update=context.user_data.get('last_update'))
            print(f"Respuesta generada: {respuesta}")

            if isinstance(respuesta, str) and respuesta.endswith(('.png', '.jpg', '.jpeg')):
                try:
                    with open(respuesta, 'rb') as img_file:
                        await update.message.reply_photo(photo=img_file)
                except FileNotFoundError:
                    await update.message.reply_text("Lo siento, no se pudo encontrar la imagen.")
            elif isinstance(respuesta, str):
                await update.message.reply_text(f"Respuesta de C-GurIN:\n{respuesta}")

            context.user_data['awaiting_voice'] = False

        await asyncio.sleep(4)

        keyboard = [
            [InlineKeyboardButton("Haz una pregunta", callback_data='ask_question')],
            [InlineKeyboardButton("Estado de la mina", callback_data='estado')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text('¡Hola! Soy C-GurIN. ¿En qué puedo ayudarte?', reply_markup=reply_markup)
    else:
        await asyncio.sleep(4)

        keyboard = [
            [InlineKeyboardButton("Haz una pregunta", callback_data='ask_question')],
            [InlineKeyboardButton("Estado de la mina", callback_data='estado')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.message.reply_text('¡Hola! Soy C-GurIN. ¿En qué puedo ayudarte?', reply_markup=reply_markup)

# Función para manejar la opción "Estado de la mina"
async def handle_estado(update: Update, context: CallbackContext):
    await update.callback_query.answer()
    df_all, last_update = update_and_return_data()
    context.user_data['last_update'] = last_update
    keyboard = [
        [InlineKeyboardButton("Ver última actualización", callback_data='last_update')],
        [InlineKeyboardButton("Ver historial del DataFrame", callback_data='dataframe')],
        [InlineKeyboardButton("Salir", callback_data='exit')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.edit_message_text(text="Estado de la mina, elige una opción:", reply_markup=reply_markup)

# Función para manejar la opción "Última actualización"
async def handle_last_update(update: Update, context: CallbackContext):
    print("Opción 'Última actualización' seleccionada")
    df_all, last_update = update_and_return_data()  # Obtener la última actualización

    # Formatear el diccionario para mostrar los valores de manera legible
    response_text = "Última actualización:\n"
    for key, value in last_update.items():
        if isinstance(value, np.float64):
            value = float(value)
        response_text += f"{key}: {value}\n"

    await update.callback_query.answer()

    await update.callback_query.edit_message_text(text=response_text)

    await asyncio.sleep(4)

    keyboard = [
        [InlineKeyboardButton("Haz una pregunta", callback_data='ask_question')],
        [InlineKeyboardButton("Estado de la mina", callback_data='estado')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text('¡Hola! Soy C-GurIN. ¿En qué puedo ayudarte?', reply_markup=reply_markup)

# Función para manejar la opción "Ver historial del DataFrame"
async def handle_dataframe(update: Update, context: CallbackContext):
    print("Opción 'Historial del DataFrame' seleccionada")
    df_all, last_update = update_and_return_data()  # Obtener el DataFrame completo

    response_text = "Historial del DataFrame (últimas filas):\n"
    formatted_df = df_all.tail().to_string(index=False, float_format="%.4f")
    response_text += formatted_df.replace("\n", "\n\n")

    await update.callback_query.answer()

    await update.callback_query.edit_message_text(text=response_text)

    await asyncio.sleep(4)

    keyboard = [
        [InlineKeyboardButton("Haz una pregunta", callback_data='ask_question')],
        [InlineKeyboardButton("Estado de la mina", callback_data='estado')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.callback_query.message.reply_text('¡Hola! Soy C-GurIN. ¿En qué puedo ayudarte?', reply_markup=reply_markup)

# Función para manejar la opción "Salir"
async def handle_exit(update: Update, context: CallbackContext):
    await update.callback_query.answer()
    await update.callback_query.edit_message_text(text="¡Hasta luego!")









