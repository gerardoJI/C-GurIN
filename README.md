# 🤖 **C-GurIN - Asistente de Seguridad para Minería Subterránea** 🤖

🚂🚂 ¡Hola! Soy Gerardo Jiménez [LinkedIn](www.linkedin.com/in/gerardo-jimenez-islas), data analyst e ingeniero de minas y metalurgia.  
Mi pasión por los datos y la inteligencia artificial me ha llevado a desarrollar **C-GurIN**, un chatbot inteligente diseñado para ser un asistente en temas de seguridad para minería subterránea.

![C-GurIN Profile](data/cgurin_profile.jpg)

## 🚨 **Descripción del Proyecto**

**C-GurIN** es un chatbot que utiliza **Telegram API**, **OpenAI**, y **Pinecone** para ofrecer respuestas en tiempo real a trabajadores de minería subterránea sobre temas de seguridad, actividades y emergencias en el entorno de la mina.

La base de datos vectorial que utiliza el sistema contiene **datos ficticios** sobre una mina metálica, generados a partir de un modelo de prueba del software **Ventsim**.

## 📂 **Estructura del Proyecto**

### 1. ⚙️ **Tecnologías y Herramientas Utilizadas**
El proyecto integra varias tecnologías clave:

- **Telegram API**: Para interactuar con los usuarios mediante un chatbot en Telegram.  
- **OpenAI (GPT)**: Para generar respuestas inteligentes a partir de las consultas de los usuarios.  
- **Pinecone**: Para realizar búsquedas eficientes en la base de datos vectorial con las preguntas de los usuarios.  
- **Wisper y Librosa**: Para la detección de voz bajo situaciones de estrés y anticipación de emergencias.  
- **Ventsim**: Para generar los datos ficticios sobre la mina, que alimentan la base de datos vectorial.

### 2. 💬 **Funciones del Chatbot**
El chatbot tiene las siguientes funcionalidades clave:

- **Ubicaciones de zonas en la mina**: El asistente responde preguntas sobre la localización de zonas específicas dentro de la mina.  
- **Planos en base64**: El chatbot puede mostrar planos de la mina codificados en formato base64.  
- **Protocolos de emergencia y actividades**: El chatbot proporciona información sobre procedimientos y actividades en situaciones de emergencia.  
- **Detección de voz bajo estrés**: A través de **Wisper** y **Librosa**, el chatbot detecta señales de estrés en la voz del usuario y puede anticipar posibles emergencias.  
- **Simulación de sensores y equipos**: El chatbot evalúa el estado de la mina en tiempo real usando datos simulados de sensores y equipos generados aleatoriamente.  
- **Multilingüismo**: El asistente puede recibir mensajes en **inglés, español, alemán, francés, italiano y portugués** (texto o audio) y responder en el mismo idioma.

### 3. 🔄 **Flujo de Funcionalidad**

1. El **usuario envía una consulta** al chatbot en Telegram (texto o audio).  
2. **Pinecone** realiza una búsqueda en la base de datos vectorial basada en la pregunta del usuario.  
3. **OpenAI (GPT)** genera una respuesta adecuada, en el idioma del usuario.  
4. Si es necesario, el chatbot puede procesar imágenes en formato base64 o detectar situaciones de estrés en la voz del usuario.  
5. El chatbot proporciona una respuesta completa y, si es necesario, información adicional sobre protocolos de emergencia, planos, o procedimientos.

## 🎥 **Demo del Proyecto**

Mira la demo del proyecto en este vídeo:  
[Demo en YouTube](https://www.youtube.com/watch?v=tORBGklYpuw)

---

## 📝 **Código de Ejemplo**

Aquí te dejo un fragmento del código principal del bot. Esta función se encarga de iniciar el bot de Telegram y agregar los manejadores para las diferentes interacciones de los usuarios:

```python
# Función principal para iniciar el bot
def main():
    application = Application.builder().token(TOKEN).build()

    # Agregar los manejadores de los botones directamente
    application.add_handler(CommandHandler('start', start))
    application.add_handler(CallbackQueryHandler(handle_last_update, pattern='^last_update$'))
    application.add_handler(CallbackQueryHandler(handle_dataframe, pattern='^dataframe$'))
    application.add_handler(CallbackQueryHandler(handle_ask_question, pattern='^ask_question$'))
    application.add_handler(MessageHandler(filters.VOICE | filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_estado, pattern='^estado$'))
    application.add_handler(CallbackQueryHandler(handle_exit, pattern='^exit$'))

    # Iniciar el bot y comenzar a escuchar los mensajes
    application.run_polling()

if __name__ == '__main__':
    main()
