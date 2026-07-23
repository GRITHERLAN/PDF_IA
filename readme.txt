# Chat PDF IA con Streamlit y LangChain

## Descripción general del proyecto

Este proyecto consiste en una aplicación web que permite cargar un archivo PDF y hacer preguntas sobre su contenido. 
La idea principal es que el usuario pueda interactuar con el documento como si estuviera conversando con él, 
obteniendo respuestas claras basadas únicamente en la información que contiene.

El sistema utiliza un enfoque de tipo RAG (Retrieval-Augmented Generation), 
lo que significa que el modelo no responde con conocimiento general, 
sino que primero busca la información relevante dentro del documento y luego construye la respuesta a partir de ese contexto.

---

## Arquitectura de la solución

La solución está organizada en varias etapas que trabajan de forma encadenada:
Primero, el usuario sube un archivo PDF desde la interfaz. 

Este archivo se procesa para extraer su contenido, 
el cual se divide en fragmentos más pequeños para facilitar su manejo.

Luego, cada fragmento se transforma en un vector numérico (embedding), 
lo que permite compararlo con otros textos en términos de similitud. Estos vectores se almacenan en una base de datos vectorial.

Cuando el usuario realiza una pregunta, el sistema busca los fragmentos del documento que más se relacionan con esa consulta. 
Con esa información se construye un contexto que se envía al modelo de lenguaje, el cual genera una respuesta basada únicamente en esos datos.

Finalmente,la respuesta se muestra en una interfaz tipo chat que mantiene el historial de la conversación.

---

## Tecnologías y herramientas utilizadas

El desarrollo del proyecto se apoya en las siguientes tecnologías:

* Streamlit, para la construcción de la interfaz web interactiva.
* LangChain, utilizado para estructurar el flujo de procesamiento y la lógica del RAG.
* PyPDFLoader, encargado de leer y extraer el contenido de los archivos PDF.
* FastEmbed, utilizado para generar los embeddings de los textos.
* ChromaDB, como base de datos vectorial para almacenar y consultar los fragmentos del documento.
* Groq API, que proporciona el modelo de lenguaje (LLaMA 3.1) encargado de generar las respuestas.

---

## Instrucciones para ejecutar el proyecto

Para ejecutar este proyecto en un entorno local, sigue estos pasos:

1. Clona el repositorio:

```bash
git clone <tu-repositorio>
cd <tu-proyecto>
```

2. Crea y activa un entorno virtual:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

4. Configura la clave de la API de Groq creando el archivo `.streamlit/secrets.toml` con el siguiente contenido:

```toml
GROQ_API_KEY = "TU_API_KEY"
```

5. Ejecuta la aplicación:

```bash
streamlit run app.py
```

---

## Ejemplos de preguntas que el agente puede responder

* ¿De qué trata el documento?
* ¿Cuál es el tema principal del PDF?
* Resume el contenido del capítulo inicial.
* ¿Qué se menciona sobre un tema específico dentro del documento?
* ¿Cuáles son las conclusiones principales?
* Explica un concepto que aparece en el texto.

---

## Ejemplos de respuestas generadas por el agente

Ejemplo 1:

Pregunta:
¿De qué trata el PDF?

Respuesta:
El documento aborda los conceptos principales relacionados con el tema desarrollado, 
explicando sus fundamentos y aplicaciones. 

A lo largo del texto se presentan definiciones, 
ejemplos y una estructura que permite comprender el contenido de manera progresiva.

---

Ejemplo 2:

Pregunta:
¿Qué dice sobre un concepto específico?

Respuesta:
El documento describe este concepto como un elemento clave dentro del tema general. 
Se explica su funcionamiento, su importancia y se incluyen ejemplos que ayudan a entender su aplicación.
[Referencia a la página correspondiente]

---

Ejemplo 3:

Pregunta:
¿Cuál es la conclusión del documento?

Respuesta:
En la parte final, el documento concluye resaltando la relevancia del tema tratado y su impacto en el contexto analizado. 
También se mencionan posibles aplicaciones y consideraciones a tener en cuenta.
[Referencia a la página correspondiente]

---

## Consideraciones

El sistema está diseñado para responder únicamente con base en la información contenida en el PDF. 
Si una pregunta no puede ser respondida con ese contenido, el sistema indicará que no encontró la información en el documento.

---

## Posibles mejoras

A futuro, este proyecto podría ampliarse incorporando funcionalidades como el soporte para múltiples documentos, 
almacenamiento persistente de la base de datos vectorial, mejoras en la calidad de recuperación de información o integración con otros formatos de archivo.
