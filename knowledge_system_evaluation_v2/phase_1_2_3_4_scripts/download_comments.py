import requests
from bs4 import BeautifulSoup
import csv
import time

# Configuramos cabeceras para simular ser un navegador
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

# --- CONFIGURACIÓN DE PAGINACIÓN ---
TOTAL_PAGINAS = 57  # Cambia este número por la cantidad de páginas que quieras raspar

# 1. Preparamos el archivo CSV
with open('mobafire_qa_paginado.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Página', 'Título', 'Pregunta Original', 'Respuestas']) # Añadí la columna 'Página' para control

    # 2. BUCLE EXTERIOR: Navegamos página por página
    for numero_pagina in range(1, TOTAL_PAGINAS + 1):
        
        # Construimos la URL dinámica según la página actual
        url_principal = f'https://www.mobafire.com/league-of-legends/questions?page={numero_pagina}'
        print(f"\n[{numero_pagina}/{TOTAL_PAGINAS}] Escaneando listado: {url_principal}")
        
        respuesta = requests.get(url_principal, headers=headers)
        soup = BeautifulSoup(respuesta.text, 'html.parser')

        # 3. Encontramos todos los bloques de preguntas de la página actual
        lista_preguntas = soup.find_all('li', class_='question-list__item')
        
        enlaces = []
        for item in lista_preguntas:
            a_tag = item.find('a')
            if a_tag and 'href' in a_tag.attrs:
                enlaces.append(a_tag['href'])

        # Si por algún motivo la página no tiene enlaces (ej. llegamos al final del foro), rompemos el bucle
        if not enlaces:
            print("No se encontraron más preguntas. Finalizando paginación.")
            break
            
        print(f"-> Se encontraron {len(enlaces)} preguntas en la página {numero_pagina}. Extrayendo...")

        # 4. BUCLE INTERIOR: Entramos a cada pregunta de esta página
        for enlace in enlaces:
            url_pregunta = "https://www.mobafire.com" + enlace
            
            resp_pregunta = requests.get(url_pregunta, headers=headers)
            soup_pregunta = BeautifulSoup(resp_pregunta.text, 'html.parser')
            
            # Extraer el TÍTULO
            info_div = soup_pregunta.find('div', class_='question-list__item__info')
            titulo = info_div.find('h4').get_text(strip=True) if info_div and info_div.find('h4') else "Sin Título"
            
            # Extraer la PREGUNTA ORIGINAL
            copy_div = soup_pregunta.find('div', class_='copy')
            pregunta = copy_div.get_text(separator=" ", strip=True) if copy_div else "Sin descripción"
            
            # Extraer las RESPUESTAS
            comentarios_divs = soup_pregunta.find_all('div', class_='comment')
            lista_textos_respuestas = []
            
            for comentario in comentarios_divs:
                contenido = comentario.find('div', class_='content')
                if contenido:
                    texto_limpio = contenido.get_text(separator=" ", strip=True)
                    lista_textos_respuestas.append(texto_limpio)
                    
            respuestas_finales = " | ".join(lista_textos_respuestas)
            
            # Escribimos la fila en el CSV
            writer.writerow([numero_pagina, titulo, pregunta, respuestas_finales])
            
            # MUY IMPORTANTE: Descanso de 2 segundos entre cada PREGUNTA
            time.sleep(2)
        
        # Descanso extra al cambiar de página principal (opcional, pero buena práctica)
        print(f"Página {numero_pagina} completada. Tomando un respiro...")
        time.sleep(3)

print("\n¡Extracción total terminada! Los datos están en mobafire_qa_paginado.csv")