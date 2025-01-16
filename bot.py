import os
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, CallbackContext, filters
from dotenv import load_dotenv
from funciones import start, handle_ask_question, handle_message, handle_estado, handle_last_update, handle_dataframe, handle_exit

# Cargar archivo .env con el api_key
load_dotenv()

# Obtener la clave API
token_chat = os.getenv("token_chatbot")
if not token_chat:
    raise ValueError("The 'token_chatbot' environment variable is not defined in the .env file.")

# Token de Telegram
TOKEN = token_chat

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
