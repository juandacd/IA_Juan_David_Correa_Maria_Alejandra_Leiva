# Actividad de Clase: Analizando Agentes de IA con Hugging Face Spaces
Maria Alejandra Leiva y Juan David Correa
## Objetivo

Explorar aplicaciones reales de Inteligencia Artificial en **Hugging
Face Spaces** y analizarlas desde la perspectiva de los **agentes
racionales**.

Al finalizar la actividad, los estudiantes deberán ser capaces de:

-   Identificar los componentes **PEAS** de un agente.
-   Clasificar las propiedades del entorno.
-   Proponer qué tipo de programa de agente podría implementarse detrás
    del sistema.
-   Justificar sus respuestas.

------------------------------------------------------------------------

## Instrucciones 

1.  Ingresen a **https://huggingface.co/spaces**.
2.  Exploren diferentes Spaces.
3.  Seleccionen uno que les parezca interesante.
4.  Interactúen con el sistema durante algunos minutos.
5.  Completen la siguiente ficha de análisis.

------------------------------------------------------------------------

# Ficha de análisis

## 1. Nombre del Space

**Nombre:** Z-Image-Turbo

**Enlace:** **https://huggingface.co/spaces/mrfakename/Z-Image-Turbo**

------------------------------------------------------------------------

## 2. ¿Qué hace el agente?

Genera una imagen a partir de un prompt escrito por el usuario. También permite ajustar parámetros como ancho, alto, semilla e inference steps.

------------------------------------------------------------------------

## 3. Análisis PEAS

  Elemento          Respuesta
  ----------------- ----------------------------------------------------
  **Performance**   ¿Qué significa que el agente haga bien su trabajo?
                     Generar una imagen que represente correctamente el prompt del usuario, con buena calidad y en poco tiempo.
  **Environment**   ¿Con qué interactúa el agente?
                    El usuario, el prompt y los parámetros de generación (ancho, alto, semilla e inference steps).
  **Actuators**     ¿Qué acciones produce?
                    Genera y muestra una imagen según la información recibida.
  **Sensors**       ¿Qué información recibe como entrada?
                     Recibe el prompt del usuario y los parámetros de generación: ancho, alto, semilla e inference steps.

------------------------------------------------------------------------

## 4. Clasificación del entorno

Complete la siguiente tabla y justifique brevemente cada respuesta.

  Propiedad      Clasificación     Justificación
  -------------- ----------------- ---------------
  **Observable**     Sí            Recibe toda la información necesaria (prompt y parámetros) para generar la imagen.
  **Determinista**   No           En cada prompt, semilla y parámetros diferentes, se generan diferentes imagenes.
  **Episódico**      Sí           Cada generación de imagen es independiente de las anteriores.
  **Estático**       Sí           El entorno no cambia mientras el agente genera la imagen.
  **Discreto**       Sí           Cada solicitud de generación es un evento independiente con una salida específica.
  **Un agente**      Sí           Solo interviene el agente generador de imágenes para realizar la tarea.
  **Conocido**       Sí           El agente conoce cómo procesar las entradas para generar la imagen.

------------------------------------------------------------------------

## 5. ¿Qué tipo de programa de agente creen que es?

**Agente basado en objetivos**.
Justificación: Su objetivo es generar una imagen que cumpla con la descripción y los parámetros proporcionados por el usuario.

------------------------------------------------------------------------

# Discusión en clase

Después de las presentaciones, discutiremos preguntas como:

-   **¿Dos Spaces diferentes pueden compartir el mismo tipo de entorno?** Sí, diferentes Spaces pueden tener el mismo tipo de entorno aunque realicen tareas distintas.
-   **¿Es posible saber con certeza qué tipo de agente implementa un Space
    únicamente observándolo?** No, porque la implementación interna no siempre es visible desde la interfaz.
-   **¿Qué diferencia existe entre el comportamiento observable de un
    agente y su implementación interna?** El comportamiento observable es lo que el agente hace, mientras que la implementación interna es cómo está programado para hacerlo.

------------------------------------------------------------------------

# Reto adicional

Encuentre un Space que pueda clasificarse como:

1.  **Totalmente observable, determinista y episódico.**: MNIST Digit Classifier
**Totalmente observable**: Recibe la imagen completa del dígito para clasificarla.
**Determinista**: La misma imagen siempre produce la misma clasificación.
**Episódico**: Cada clasificación es independiente de las anteriores.
3.  **Parcialmente observable, estocástico y secuencial.**: Text Adventure
**Parcialmente observable:** El agente solo conoce la información que el jugador ha revelado durante la partida.
**Estocástico**: Las respuestas pueden cambiar aunque el jugador escriba el mismo comando.
**Secuencial**: Cada acción afecta el estado del juego y las decisiones futuras.
------------------------------------------------------------------------

# Rúbrica (10 puntos)

| Criterio | Puntos |
|-----------|:------:|
| Descripción correcta del Space | 2 |
| Identificación de PEAS | 3 |
| Clasificación del entorno | 3 |
| Justificación del tipo de agente | 2 |
| **Total** | **10** |

------------------------------------------------------------------------


