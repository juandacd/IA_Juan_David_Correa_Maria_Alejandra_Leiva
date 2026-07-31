# Actividad de Clase: Analizando Agentes de IA con Hugging Face Spaces

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

Modifica las imagenes cargadas segun las intrucciones escritas, generando nuevas imagenes o videos

------------------------------------------------------------------------

## 3. Análisis PEAS

  Elemento          Respuesta
  ----------------- ----------------------------------------------------
  **Performance**   ¿Qué significa que el agente haga bien su trabajo?
                      Realize de forma correcta los cambios y propuestas sugeridas por el usuario generando de forma correcta la imagen o video propuesto usando                         la imagen cargada
  **Environment**   ¿Con qué interactúa el agente?
                    interactua virtualmente con el usuario y la imagen cargada
  **Actuators**     ¿Qué acciones produce?
                    crea una imagen o video segun las condiciones del usauario y la imagen proporcionada
  **Sensors**       ¿Qué información recibe como entrada?
                      Recibe una imagen y un prompt

------------------------------------------------------------------------

## 4. Clasificación del entorno

Complete la siguiente tabla y justifique brevemente cada respuesta.

  Propiedad      Clasificación     Justificación
  -------------- ----------------- ---------------
  Observable     Total               el agente tiene acceso total a la informacion proporcionada y al entorno
  Determinista   No                   Los modelos generativos utilizados para editar imágenes incluyen procesos probabilísticos, por lo cual no genera siempre las                                       mismas repuestas
  Episódico      No           no se mueve ni cambia con los prcesos anteriores
  Estático       Sí           no se ve afectado por escritos anteriores
  Discreto       No           como son imagenes y videos credos se representa mediante valores continuos
  Conocido       Sí           

------------------------------------------------------------------------

## 5. ¿Qué tipo de programa de agente creen que es?

Seleccione la opción que consideren más adecuada y explique por qué.
-   Agente basado en objetivos: se base en cumplir el prompt realizado por el usuario
  
-   Agente con aprendizaje: ya que de lo escrito y quejas realiza mejores resultados, aunque me entra la duda que no usa resultados de prompt anteriores

------------------------------------------------------------------------

# Discusión en clase

Después de las presentaciones, discutiremos preguntas como:

-   ¿Dos Spaces diferentes pueden compartir el mismo tipo de entorno? si
-   ¿Es posible saber con certeza qué tipo de agente implementa un Space
    únicamente observándolo? no
-   ¿Qué diferencia existe entre el comportamiento observable de un
    agente y su implementación interna? los recursos o informacion adicional que use el agente

------------------------------------------------------------------------

# Reto adicional

Encuentre un Space que pueda clasificarse como:

1.  **Totalmente observable, determinista y episódico.**: MNIST Digit Classifier                                                                                   Totalmente observable: la imagen completa del dígito es visible para el agente.
Determinista: para una misma imagen y un mismo modelo, el resultado siempre es el mismo.
Episódico: cada clasificación de una imagen es independiente de las anteriores; no existe memoria entre episodios.
2.  **Parcialmente observable, estocástico y secuencial.**: Text Adventure (LLM Game)
Parcialmente observable: el agente solo conoce la descripción textual del estado actual, no el estado completo del mundo.
Estocástico: las respuestas del modelo de lenguaje pueden variar incluso ante la misma acción, introduciendo incertidumbre.
Secuencial: cada decisión afecta el estado futuro del juego y las opciones disponibles.


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


