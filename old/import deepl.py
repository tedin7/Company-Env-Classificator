import pandas as pd
from googletrans import Translator
from tqdm import tqdm
import re

# Initialize translator
translator = Translator()

def translate_to_english(text):
    if not text.strip():  # Skip translation if text is empty or whitespace
        return text
    
    try:
        # Automatically detect the language of the source text
        translation = translator.translate(text, dest='en')
        return translation.text
    except Exception as e:
        print(f"Error during translation: {e}")
        return text

# Example usage
if __name__ == '__main__':
    text = "Somos un grupo financiero independiente especializado en la gestión de Patrimonios desde 1984 Nuestro grupo lo componen DUX INVERSORES CAPITAL AV Y DUX INVERSORES SGIIC Desde nuestros orígenes hemos tratado de ofrecer un servicio financiero personalizado con la confianza y profesionalidad como principios fundamentales Nuestro trabajo es vocacional Nos gusta lo que hacemos y eso se traslada a los resultados de nuestros clientes Gestionamos con autonomía y libertad al no estar subordinados a ningún otro grupo financiero u objetivos comerciales Nuestro EQUIPO cuenta con una alta cualificación, acreditaciones CFA y EFPA y experiencia previa en el sector financiero Contamos con una experiencia de más de 30 años adoptando siempre los más altos estándares éticos Los SOCIOS se encuentran involucrados en la actividad diaria y en la relación directa con los clientes invertimos en los mismos productos que nuestros clientes Lo mejor para tí es lo mejor para nosotros Contamos con la colaboración de Entidades de reconocido prestigio en el sector y nos encontramos permanentemente supervisados por diferentes organismos públicos Adoptamos un modelo Fintech buscando nuevas fórmulas y opciones para gestionar y relacionarnos con nuestros clientes"
    translated_text = translate_to_english(text)
    print("Original:", text)
    print("Translated:", translated_text)
