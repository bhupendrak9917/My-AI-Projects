{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "mnist_streamlit.ipynb",
      "provenance": [],
      "mount_file_id": "1ca8IKDoz1gP1yoL2g1IbJLLbEFShMiWL",
      "authorship_tag": "ABX9TyPUxHOgFyUmi87QKYlvmoz/",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bhupendrak9917/My-AI-Projects/blob/main/MNIST_Streamlit/mnist_streamlit.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "z9gDlju6bxch",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "451e5a6c-c956-46ff-8bb0-047341c2245f"
      },
      "source": [
        "!pip install streamlit --quiet\n",
        "!pip install pyngrok==4.1.1 --quiet\n",
        "!pip install streamlit-drawable-canvas --quiet\n",
        "from pyngrok import ngrok"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\u001b[K     |████████████████████████████████| 8.2MB 4.0MB/s \n",
            "\u001b[K     |████████████████████████████████| 174kB 38.7MB/s \n",
            "\u001b[K     |████████████████████████████████| 81kB 9.4MB/s \n",
            "\u001b[K     |████████████████████████████████| 4.2MB 27.2MB/s \n",
            "\u001b[K     |████████████████████████████████| 112kB 47.5MB/s \n",
            "\u001b[K     |████████████████████████████████| 71kB 8.9MB/s \n",
            "\u001b[K     |████████████████████████████████| 122kB 60.1MB/s \n",
            "\u001b[?25h  Building wheel for blinker (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[31mERROR: google-colab 1.0.0 has requirement ipykernel~=4.10, but you'll have ipykernel 5.5.5 which is incompatible.\u001b[0m\n",
            "  Building wheel for pyngrok (setup.py) ... \u001b[?25l\u001b[?25hdone\n",
            "\u001b[K     |████████████████████████████████| 1.3MB 3.9MB/s \n",
            "\u001b[?25h"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MHrVpXOdhBbM",
        "outputId": "c28eae83-a04f-4235-9a5b-f8e194959b43"
      },
      "source": [
        "%%writefile app.py\n",
        "import streamlit as st\n",
        "from streamlit_drawable_canvas import st_canvas\n",
        "from tensorflow import keras\n",
        "import cv2\n",
        "import numpy as np\n",
        "model_new = keras.models.load_model('/content/drive/MyDrive/Minor Project <Bhupendra Kumar>/mnist.hdf5')\n",
        "\n",
        "st.title(\"MNIST Digit Recognizer\")\n",
        "\n",
        "SIZE = 192\n",
        "\n",
        "canvas_result = st_canvas(\n",
        "    fill_color=\"#ffffff\",\n",
        "    stroke_width=10,\n",
        "    stroke_color='#ffffff',\n",
        "    background_color=\"#000000\",\n",
        "    height=150,width=150,\n",
        "    drawing_mode='freedraw',\n",
        "    key=\"canvas\",\n",
        ")\n",
        "\n",
        "if canvas_result.image_data is not None:\n",
        "    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))\n",
        "    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)\n",
        "    st.write('Input Image')\n",
        "    st.image(img_rescaling)\n",
        "\n",
        "if st.button('Predict'):\n",
        "    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)\n",
        "    pred = model_new.predict(test_x.reshape(1, 28, 28, 1))\n",
        "    st.write(f'result: {np.argmax(pred[0])}')\n",
        "    st.bar_chart(pred[0])\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Writing app.py\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 52
        },
        "id": "v8nQqxZ-l5Dc",
        "outputId": "99a29bef-9053-4913-8600-eb73768450e7"
      },
      "source": [
        "!nohup streamlit run app.py &\n",
        "url = ngrok.connect(port='8501')\n",
        "url"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "nohup: appending output to 'nohup.out'\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'http://2ab00b5f1a5e.ngrok.io'"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xB-aBq0Tl9Ct"
      },
      "source": [
        ""
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}