{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Assessment1Task2 - lr0.0001.ipynb",
      "provenance": [],
      "collapsed_sections": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qPOTFVMKxNf3"
      },
      "source": [
        "#CS5079 -  Assessment 1\n",
        "##Task 2\n",
        "\n",
        "The code in this document is largely based on Tutorial 3, and also includes some code from https://colab.research.google.com/drive/18LdlDDT87eb8cCTHZsXyS9ksQPzL3i6H for attempting to render the environment in Google Colab."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "hggDzu4gxtLq"
      },
      "source": [
        "Importing the required packages and libraries for this task"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "rf37FxFUGijf",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "effb05c7-70dd-4bb0-c20d-f783436de69b"
      },
      "source": [
        "!pip install gym\n",
        "!apt-get install python-opengl -y\n",
        "!apt install xvfb -y"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: gym in /usr/local/lib/python3.6/dist-packages (0.17.3)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from gym) (1.4.1)\n",
            "Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from gym) (1.3.0)\n",
            "Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from gym) (1.5.0)\n",
            "Requirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.6/dist-packages (from gym) (1.18.5)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from pyglet<=1.5.0,>=1.4.0->gym) (0.16.0)\n",
            "Reading package lists... Done\n",
            "Building dependency tree       \n",
            "Reading state information... Done\n",
            "python-opengl is already the newest version (3.1.0+dfsg-1).\n",
            "0 upgraded, 0 newly installed, 0 to remove and 14 not upgraded.\n",
            "Reading package lists... Done\n",
            "Building dependency tree       \n",
            "Reading state information... Done\n",
            "xvfb is already the newest version (2:1.19.6-1ubuntu4.8).\n",
            "0 upgraded, 0 newly installed, 0 to remove and 14 not upgraded.\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "onmyFCmUGn98",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "15aef5ec-86f2-4ac5-a5f1-50cb18619db3"
      },
      "source": [
        "!pip install gym[atari]"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: gym[atari] in /usr/local/lib/python3.6/dist-packages (0.17.3)\n",
            "Requirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.18.5)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.4.1)\n",
            "Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.5.0)\n",
            "Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.3.0)\n",
            "Requirement already satisfied: Pillow; extra == \"atari\" in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (7.0.0)\n",
            "Requirement already satisfied: opencv-python; extra == \"atari\" in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (4.1.2.30)\n",
            "Requirement already satisfied: atari-py~=0.2.0; extra == \"atari\" in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (0.2.6)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from pyglet<=1.5.0,>=1.4.0->gym[atari]) (0.16.0)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from atari-py~=0.2.0; extra == \"atari\"->gym[atari]) (1.15.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Pw75vipGGo4o",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "77a8f515-b44f-4753-dd1d-2c4358603004"
      },
      "source": [
        "!pip install pyvirtualdisplay\n",
        "!pip install piglet"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: pyvirtualdisplay in /usr/local/lib/python3.6/dist-packages (1.3.2)\n",
            "Requirement already satisfied: EasyProcess in /usr/local/lib/python3.6/dist-packages (from pyvirtualdisplay) (0.3)\n",
            "Requirement already satisfied: piglet in /usr/local/lib/python3.6/dist-packages (1.0.0)\n",
            "Requirement already satisfied: piglet-templates in /usr/local/lib/python3.6/dist-packages (from piglet) (1.0.0)\n",
            "Requirement already satisfied: markupsafe in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (1.1.1)\n",
            "Requirement already satisfied: attrs in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (20.3.0)\n",
            "Requirement already satisfied: astunparse in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (1.6.3)\n",
            "Requirement already satisfied: Parsley in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (1.3)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.6/dist-packages (from astunparse->piglet-templates->piglet) (0.35.1)\n",
            "Requirement already satisfied: six<2.0,>=1.6.1 in /usr/local/lib/python3.6/dist-packages (from astunparse->piglet-templates->piglet) (1.15.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "yIJIdj2wGr9M",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "085148c4-6d1e-4931-d4b1-ee9d4d2ff9c4"
      },
      "source": [
        "from pyvirtualdisplay import Display\n",
        "display = Display(visible=0, size=(1400, 900))\n",
        "display.start()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<pyvirtualdisplay.display.Display at 0x7fa8c177a390>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 21
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ldlaF2QeGvkc"
      },
      "source": [
        "# This code creates a virtual display to draw game images on. \n",
        "# If you are running locally, just ignore it\n",
        "import os\n",
        "if type(os.environ.get(\"DISPLAY\")) is not str or len(os.environ.get(\"DISPLAY\"))==0:\n",
        "    !bash ../xvfb start\n",
        "    %env DISPLAY=:1"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "7k41Nx-okbyn"
      },
      "source": [
        "import gym\n",
        "from gym import logger as gymlogger\n",
        "from gym.wrappers import Monitor\n",
        "gymlogger.set_level(40) # error only\n",
        "import tensorflow.compat.v1 as tf\n",
        "import numpy as np\n",
        "import random\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "import os\n",
        "import math\n",
        "import glob\n",
        "import io\n",
        "import base64\n",
        "from gym import wrappers\n",
        "import operator\n",
        "from IPython.display import HTML\n",
        "\n",
        "from IPython import display as ipythondisplay"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "PoE7nEenx07N"
      },
      "source": [
        "Creating a function to show the video of the Atari game environment"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dhyPXqZYkj9j"
      },
      "source": [
        "\"\"\"\n",
        "Utility functions to enable video recording of gym environment and displaying it\n",
        "To enable video, just do \"env = wrap_env(env)\"\"\n",
        "\"\"\"\n",
        "\n",
        "def show_video():\n",
        "  mp4list = glob.glob('video/*.mp4')\n",
        "  if len(mp4list) > 0:\n",
        "    mp4 = mp4list[0]\n",
        "    video = io.open(mp4, 'r+b').read()\n",
        "    encoded = base64.b64encode(video)\n",
        "    ipythondisplay.display(HTML(data='''<video alt=\"test\" autoplay \n",
        "                loop controls style=\"height: 400px;\">\n",
        "                <source src=\"data:video/mp4;base64,{0}\" type=\"video/mp4\" />\n",
        "             </video>'''.format(encoded.decode('ascii'))))\n",
        "  else: \n",
        "    print(\"Could not find video\")\n",
        "    \n",
        "\n",
        "def wrap_env(env):\n",
        "  env = Monitor(env, './video', force=True)\n",
        "  return env"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ls2AhFW3x6uD"
      },
      "source": [
        "Setting the environment being used as the RAM version of Seaquest, and showing a sample output of the environment"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "D22tjxmzke-E",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0742a4e0-7eee-46d5-e418-460554882c81"
      },
      "source": [
        "env = gym.make(\"Seaquest-ram-v0\")\n",
        "env.reset()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([  0,  66, 204,   0,  15,   0, 255,  26, 175, 144,  24, 170, 132,\n",
              "         0,  12,   6,  50, 134, 212, 253,   0, 253,  86, 253, 164, 253,\n",
              "        80, 254,   0, 254,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
              "         0,   0,   0,   0,   0, 200, 200, 200, 200,   0,   1,   2,   3,\n",
              "       255, 255, 255,   0,   0,   0,   0,   3,   0,   0,   0,   0,   0,\n",
              "         0,   0,   0,   0,   0,  76,   0,   0,   0,   0, 101,  96,  48,\n",
              "        32,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
              "         0,   0,   0,   0,   0,   0,  13,   0, 255, 255,   0,   0,   0,\n",
              "         1,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,\n",
              "         0,   0,  96,   7,   0,   0,  85, 215, 254, 214, 244], dtype=uint8)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 25
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EEFWBBcbyFZb"
      },
      "source": [
        "Creating a graph to show which bytes of RAM may have a larger impact while the game is running"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "id": "T0K3V6w5vjYR",
        "outputId": "7587d9f6-5e5f-4c59-ba92-f56e5398f895"
      },
      "source": [
        "observation_image, reward, done, info = env.step(0)\n",
        "observation_ram = env.unwrapped._get_ram()\n",
        "env = env.unwrapped\n",
        "env.reset()\n",
        "\n",
        "plt.title('RAM Inputs Visualization',size=12)\n",
        "plt.xlabel('Bytes (0-127)')\n",
        "plt.ylabel('Array Values (0-255)')\n",
        "plt.plot(observation_ram)\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Rl3BLsvvyH_n"
      },
      "source": [
        "Filtering the RAM inputs which have average values over 100, to see which RAM inputs will have the biggest impact."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5KEwZwgYGDAW",
        "outputId": "32010af0-86e2-4568-a356-355186cd1b09"
      },
      "source": [
        "temp=[]\n",
        "temp_new = []\n",
        "for i in range(0,127):\n",
        "  if observation_ram[i]> 100:\n",
        "    temp.append(observation_ram[i])\n",
        "    temp_new.append(observation_ram[i])\n",
        "  else:\n",
        "    temp.append(0)\n",
        "  \n",
        "print(temp)  \n",
        "np.array([temp_new]).shape"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "[0, 0, 204, 0, 0, 0, 255, 0, 175, 144, 0, 170, 132, 0, 0, 0, 0, 134, 212, 253, 0, 253, 0, 253, 192, 253, 0, 254, 0, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 200, 200, 200, 0, 0, 0, 0, 255, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 101, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 215, 254, 214]\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "(1, 28)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 27
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "i0V4BWOGF0vz",
        "outputId": "1d7741ca-57cc-4d6e-ef35-a43ab284e703"
      },
      "source": [
        "print(\"Maximum Value:    \", observation_ram.max())\n",
        "print(\"Minimum Value:    \", observation_ram.min())\n",
        "print(\"Mean height:       \", observation_ram.mean())\n",
        "print(\"Standard deviation:\", observation_ram.std())"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Maximum Value:     255\n",
            "Minimum Value:     0\n",
            "Mean height:        55.2890625\n",
            "Standard deviation: 91.30735802973983\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "dnnCD0wvyV54"
      },
      "source": [
        "Showing the observation and action spaces of the environment, to know what the dimensions of the output and input are"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "zXpp4C7yDQT2",
        "outputId": "365710fb-d31f-47a1-8122-03e633a429d0"
      },
      "source": [
        "print(\"Observation space:\", env.observation_space)\n",
        "print(\"Action space:\", env.action_space)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Observation space: Box(0, 255, (128,), uint8)\n",
            "Action space: Discrete(18)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "p8DSrGjPyYOD"
      },
      "source": [
        "Creating a random agent to see what an average expected reward should be prior to training."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "yiG8HUadDVbs",
        "outputId": "eb1de570-6961-4274-c3e3-64ba613a9276"
      },
      "source": [
        "class RandomAgent():\n",
        "    def __init__(self, env):\n",
        "        self.action_size = env.action_space.n\n",
        "        \n",
        "    def get_action(self, observation):\n",
        "        return random.choice(range(self.action_size))\n",
        "    \n",
        "total_reward=0\n",
        "agent = RandomAgent(env)\n",
        "numberOfEpisodes = 10\n",
        "for steps in range(numberOfEpisodes):\n",
        "    current_obs = env.reset()\n",
        "    done = True\n",
        "    while not done:\n",
        "        action = agent.get_action(current_obs)\n",
        "        next_obs, reward, done, info = env.step(action)\n",
        "        total_reward += reward\n",
        "        env.render()\n",
        "print(\"Average reward: {}\".format(total_reward/numberOfEpisodes))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Average reward: 0.0\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "UMaRIuaWynwh"
      },
      "source": [
        "Resetting the environment after testing the agent"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "BiyyvsQc7Nrk"
      },
      "source": [
        "obs = env.reset()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "FIsOnyp8eRJc"
      },
      "source": [
        "Q Network architecture based on http://cs229.stanford.edu/proj2016/report/BonillaZengZheng-AsynchronousDeepQLearningforBreakout-Report.pdf, and adapted from Tutorial 3"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "jHUHv4JeDhR1"
      },
      "source": [
        "def q_network(X_state, name):\n",
        "    prev_layer = X_state/255 # scale pixel intensities to the [-1.0, 1.0] range.\n",
        "    initializer = tf.variance_scaling_initializer()\n",
        "    with tf.variable_scope(name) as scope:\n",
        "        prev_layer = tf.reshape(prev_layer, shape=[1, 128])\n",
        "        prev_layer = tf.layers.dense(prev_layer,128,\n",
        "                                 activation=tf.nn.relu,\n",
        "                                 kernel_initializer=initializer)\n",
        "        prev_layer = tf.layers.dense(prev_layer,128,\n",
        "                                 activation=tf.nn.relu,\n",
        "                                 kernel_initializer=initializer)\n",
        "        prev_layer = tf.layers.dense(prev_layer,128,\n",
        "                                 activation=tf.nn.relu,\n",
        "                                 kernel_initializer=initializer)\n",
        "        hidden = tf.layers.dense(prev_layer,6,\n",
        "                                 activation=tf.nn.relu,\n",
        "                                 kernel_initializer=initializer)\n",
        "        outputs = tf.layers.dense(hidden, env.action_space.n,kernel_initializer=initializer)\n",
        "    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)\n",
        "    trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}\n",
        "    return outputs, trainable_vars_by_name"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "XTQlII8Lyz9_"
      },
      "source": [
        "Creating the QLearningAgent class, adapted from Tutorial 3"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Moo0Qi2nDxPS"
      },
      "source": [
        "class QLearningAgent():\n",
        "    def __init__(self, env, learning_rate = 0.0001, momentum = 0.95):\n",
        "        self.loss_val = np.infty\n",
        "        self.action_size = env.action_space.n\n",
        "        tf.reset_default_graph()\n",
        "        tf.disable_eager_execution()\n",
        "        self.discount_rate = 0.99\n",
        "        self.checkpoint_path = \"./my_dqn.ckpt\"\n",
        "        self.X_state = tf.placeholder(tf.float32, shape=[1, 128]) #the RAM state\n",
        "        self.online_q_values, self.online_vars = q_network(self.X_state, name=\"q_networks/online\")\n",
        "        self.target_q_values, self.target_vars = q_network(self.X_state, name=\"q_networks/target\")\n",
        "\n",
        "        #The \"target\" DNN will take the values of the \"online\" DNN\n",
        "        self.copy_ops = [target_var.assign(self.online_vars[var_name]) for var_name, target_var in self.target_vars.items()]\n",
        "        self.copy_online_to_target = tf.group(*self.copy_ops)\n",
        "\n",
        "        #We create the model for training\n",
        "        with tf.variable_scope(\"train\"):\n",
        "            self.X_action = tf.placeholder(tf.int32, shape=[None])\n",
        "            self.y = tf.placeholder(tf.float32, shape=[None, 1])\n",
        "            self.q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.action_size),axis=1, keepdims=True)\n",
        "            \n",
        "            #If the error is between 0 and 1, \n",
        "            self.error = tf.abs(self.y - self.q_value)\n",
        "            self.clipped_error = tf.clip_by_value(self.error, 0.0, 1.0)\n",
        "            self.linear_error = 2 * (self.error - self.clipped_error)\n",
        "            self.loss = tf.reduce_mean(tf.square(self.clipped_error) + self.linear_error)\n",
        "            \n",
        "            \n",
        "            \n",
        "            self.global_step = tf.Variable(0, trainable=False, name='global_step')\n",
        "            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)\n",
        "            self.training_op = self.optimizer.minimize(self.loss, global_step=self.global_step)\n",
        "\n",
        "        # Saving the session\n",
        "        self.saver = tf.train.Saver()\n",
        "        self.sess = tf.Session()\n",
        "        if os.path.isfile(self.checkpoint_path + \".index\"):\n",
        "            self.saver.restore(self.sess, self.checkpoint_path)\n",
        "        else:\n",
        "            self.sess.run(tf.global_variables_initializer())\n",
        "            self.sess.run(self.copy_online_to_target)\n",
        "        \n",
        "    #---- CHOSSING ACTION ----\n",
        "    def get_action(self,q_values, step):\n",
        "        epsilon = max(0.1, 1 - (0.9/2000000) * step)\n",
        "        if np.random.rand() < epsilon:\n",
        "            return np.random.randint(self.action_size) # random action\n",
        "        else:\n",
        "            return np.argmax(q_values) # optimal action\n",
        "\n",
        "    #---- TRAINING ----\n",
        "    def train(self, state_val, action_val, reward, next_state_val, continues):\n",
        "        # Compute next_qvalues  \n",
        "        next_q_values = self.target_q_values.eval(feed_dict={self.X_state: np.array([next_state_val])})\n",
        "        # Compute best rewards\n",
        "        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)\n",
        "        # Compute target values\n",
        "        y_val = reward + continues * self.discount_rate * max_next_q_values\n",
        "        # Train the online DQN\n",
        "        _, self.loss_val = self.sess.run([self.training_op, self.loss], feed_dict={self.X_state: np.array([state_val]), self.X_action: np.array([action_val]), self.y: y_val})"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "zBkxYMzhy6Rr"
      },
      "source": [
        "Setting up the agent, and establishing lists to store the loss, steps per episode, and the step rewards.\n",
        "\n",
        "The number of steps has been set to 1,000,000, with a batch size of 25."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oWJkJEZmD2gH",
        "outputId": "ef48dac7-1fcc-474a-ce07-d7239d1db7b7"
      },
      "source": [
        "agent = QLearningAgent(env)  \n",
        "ep_rewards = []\n",
        "step_loss=[]\n",
        "total_reward = 0\n",
        "steps_per_episode=[]\n",
        "steps_counter=0\n",
        "step_rewards=[]\n",
        "n_steps = 1000000  # total number of training steps\n",
        "copy_steps = 5000\n",
        "save_steps = 1000\n",
        "done=True\n",
        "batch_size=25 \n",
        "\n",
        "with agent.sess:\n",
        "    while True:\n",
        "        step = agent.global_step.eval()\n",
        "        if step >= n_steps:\n",
        "            break\n",
        "\n",
        "        print(\"\\r\\tTraining step {}/{} ({:.1f})%\\tLoss {:5f}\".format(\n",
        "            step,\n",
        "            n_steps,\n",
        "            step * 100 / n_steps, \n",
        "            agent.loss_val), end=\"\")\n",
        "\n",
        "        if done: # game over, start again\n",
        "            obs = env.reset()\n",
        "            ep_rewards.append(total_reward)\n",
        "            steps_per_episode.append(steps_counter)\n",
        "            steps_counter=0\n",
        "            total_reward = 0\n",
        "            state = obs #Removed preprocessing as no image input\n",
        "\n",
        "        total_perc = int(step * 100 / n_steps)\n",
        "        \n",
        "        # Online DQN evaluates what to do\n",
        "        q_values = agent.online_q_values.eval(feed_dict={agent.X_state: [state]})\n",
        "        action = agent.get_action(q_values, step)\n",
        "        \n",
        "        # Online DQN plays\n",
        "        next_obs, reward, done, info = env.step(action)\n",
        "        next_state = next_obs\n",
        "\n",
        "        agent.train(state, action, reward, next_state, 1.0 - done)\n",
        "        \n",
        "        #env.render()\n",
        "        total_reward+=reward\n",
        "        steps_counter+=1\n",
        "        step_loss.append(agent.loss_val)\n",
        "        step_rewards.append(reward)\n",
        "        state = next_state\n",
        "\n",
        "        # Regularly copy the online DQN to the target DQN\n",
        "        if step % copy_steps == 0:\n",
        "            agent.copy_online_to_target.run()\n",
        "\n",
        "        # And save regularly\n",
        "        if step % save_steps == 0:\n",
        "            agent.saver.save(agent.sess, agent.checkpoint_path)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\tTraining step 999999/1000000 (100.0)%\tLoss 0.000001"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gWCgMHjSzIsr"
      },
      "source": [
        "Printing the number of steps, and the number of episodes after training"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "TF2VWW_XcB6j",
        "outputId": "998cc63b-fa4f-48ea-98bf-567c8de2ffa6"
      },
      "source": [
        "len(step_loss)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1000000"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 35
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aMFUZz0scFeU",
        "outputId": "a5383c2e-52c1-4471-8466-7cc66a99f2c0"
      },
      "source": [
        "len(ep_rewards)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1201"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 36
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "_oI1z659zPeb"
      },
      "source": [
        "Plotting the number of episodes against the rewards\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "wEoKjMwVEsi9",
        "outputId": "0d82f463-8803-4a1c-a933-77b829293331"
      },
      "source": [
        "plt.plot(range(len(ep_rewards)), ep_rewards)\n",
        "plt.xlabel('Episodes')\n",
        "plt.ylabel('Rewards')\n",
        "plt.savefig('images/ep_reward.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 37
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "25Vwym3KzVLB"
      },
      "source": [
        "Plotting the step loss against the number of steps"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "6m-z9VIobGOT",
        "outputId": "872155f5-1a68-4c2f-ec04-3d9016744fc2"
      },
      "source": [
        "plt.plot(range(len(step_loss)), step_loss)\n",
        "plt.xlabel('Steps')\n",
        "plt.ylabel('Loss')\n",
        "plt.savefig('images/step_loss.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 38
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wvMsHCGAzi1g"
      },
      "source": [
        "Creating graphs to show averaged rewards for each episode, averaged for every 100 episodes, and averaged graphs for loss on each step, averaged over 1000, 10,000 and 100,000 steps to detect improvement on a larger scale."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "id": "tWFIB9dmbI9H",
        "outputId": "6fccd451-f56b-4ce7-9318-e05b922c85ee"
      },
      "source": [
        "avg=100\n",
        "ep_rewards_avg=[np.mean(ep_rewards[i:i+avg]) for i in range(0,len(ep_rewards),avg)]\n",
        "\n",
        "plt.plot(range(len(ep_rewards_avg)), ep_rewards_avg)\n",
        "plt.xlabel('Episodes')\n",
        "plt.ylabel('Rewards')\n",
        "plt.savefig('images/ep_reward_avg_100.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 39
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "id": "mLe11VsebMdI",
        "outputId": "cc687d03-ba10-4926-be14-2367981ab16a"
      },
      "source": [
        "avg=1000\n",
        "step_loss_avg=[np.mean(step_loss[i:i+avg]) for i in range(0,len(step_loss),avg)]\n",
        "print(len(step_loss_avg))\n",
        "plt.plot(range(len(step_loss_avg)), step_loss_avg)\n",
        "plt.xlabel('Steps')\n",
        "plt.xticks(range(0,len(step_loss_avg)),range(0,len(step_loss),avg))\n",
        "plt.ylabel('Loss average')\n",
        "plt.savefig('images/step_loss_avg_1000.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "1000\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 40
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dJSgNuL2i8B4",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 315
        },
        "outputId": "0aa2a358-b65a-4c18-d120-5f3cfe3114a7"
      },
      "source": [
        "avg=10000\n",
        "step_loss_avg=[np.mean(step_loss[i:i+avg]) for i in range(0,len(step_loss),avg)]\n",
        "print(len(step_loss_avg))\n",
        "plt.plot(range(len(step_loss_avg)), step_loss_avg)\n",
        "plt.xlabel('Steps')\n",
        "plt.xticks(range(0,len(step_loss_avg)),range(0,len(step_loss),avg))\n",
        "plt.ylabel('Loss average')\n",
        "plt.savefig('images/step_loss_avg_10000.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "100\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 41
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "A6taIPOVi7zI",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "outputId": "098dbae7-a462-44ba-cfca-04aa6d5b0254"
      },
      "source": [
        "avg=100000\n",
        "step_loss_avg=[np.mean(step_loss[i:i+avg]) for i in range(0,len(step_loss),avg)]\n",
        "print(len(step_loss_avg))\n",
        "plt.plot(range(len(step_loss_avg)), step_loss_avg)\n",
        "plt.xlabel('Steps')\n",
        "plt.xticks(range(0,len(step_loss_avg)),range(0,len(step_loss),avg))\n",
        "plt.ylabel('Loss average')\n",
        "plt.savefig('images/step_loss_avg_100000.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "10\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 42
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "lnyFzccnzwgk"
      },
      "source": [
        "Plotting the number of steps per episode on an aggregated basis for every 100 episodes, to see the changes in the number of steps over time"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 315
        },
        "id": "lt0SWkCWbOS1",
        "outputId": "f040e40a-7653-48aa-beb3-d9d5e84e8322"
      },
      "source": [
        "avg=100\n",
        "steps_per_episode_avg=[np.mean(steps_per_episode[i:i+avg]) for i in range(0,len(steps_per_episode),avg)]\n",
        "print(len(step_loss_avg))\n",
        "\n",
        "plt.plot(range(len(steps_per_episode_avg)), steps_per_episode_avg)\n",
        "plt.xticks(range(0,len(steps_per_episode_avg)),range(0,len(steps_per_episode),avg))\n",
        "plt.xlabel('Episodes')\n",
        "plt.ylabel('Steps in episode')\n",
        "plt.savefig('images/step_per_episode_100.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "10\n"
          ],
          "name": "stdout"
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 43
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    }
  ]
}