{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.7.6"
    },
    "colab": {
      "name": "Assessment1Task1withExperienceBonus.ipynb",
      "provenance": [],
      "toc_visible": true
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "BiAmF-cXkj_5"
      },
      "source": [
        "CS5079 Assessment 1 Task 1 (with Experience) with the bonus task of frame skipping"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "kkpoj4c9HT7I"
      },
      "source": [
        "# import libraries that are required to build the agent and train \n",
        "import tensorflow.compat.v1 as tf\n",
        "import numpy as np\n",
        "import random\n",
        "import os\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "import gym\n",
        "env = gym.make(\"Seaquest-v0\")"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "r3DmY2aVghk1"
      },
      "source": [
        "#from collections import deque"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bSSv7OUitjrV",
        "outputId": "f18090c5-ef71-4a20-ad69-a209614e9a96"
      },
      "source": [
        "for i in env.get_action_meanings():\n",
        "  print(i)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "NOOP\n",
            "FIRE\n",
            "UP\n",
            "RIGHT\n",
            "LEFT\n",
            "DOWN\n",
            "UPRIGHT\n",
            "UPLEFT\n",
            "DOWNRIGHT\n",
            "DOWNLEFT\n",
            "UPFIRE\n",
            "RIGHTFIRE\n",
            "LEFTFIRE\n",
            "DOWNFIRE\n",
            "UPRIGHTFIRE\n",
            "UPLEFTFIRE\n",
            "DOWNRIGHTFIRE\n",
            "DOWNLEFTFIRE\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "TjaeH3G-1aFW",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ae21099d-bda5-453e-f112-be16601230a2"
      },
      "source": [
        "# This code allows the creation of a video file despite the rendering error on google colab\n",
        "# code taken from: https://colab.research.google.com/drive/18LdlDDT87eb8cCTHZsXyS9ksQPzL3i6H\n",
        "\n",
        "!pip install gym\n",
        "!apt-get install python-opengl -y\n",
        "!apt install xvfb -y\n",
        "!pip install gym[atari]\n",
        "!pip install pyvirtualdisplay\n",
        "!pip install piglet\n",
        "\n",
        "from pyvirtualdisplay import Display\n",
        "display = Display(visible=0, size=(1400, 900))\n",
        "display.start()\n",
        "import os\n",
        "if type(os.environ.get(\"DISPLAY\")) is not str or len(os.environ.get(\"DISPLAY\"))==0:\n",
        "    !bash ../xvfb start\n",
        "    %env DISPLAY=:1\n",
        "\n",
        "import gym\n",
        "from gym import logger as gymlogger\n",
        "from gym.wrappers import Monitor\n",
        "gymlogger.set_level(40) # error only\n",
        "\n",
        "import numpy as np\n",
        "import random\n",
        "import matplotlib\n",
        "import matplotlib.pyplot as plt\n",
        "%matplotlib inline\n",
        "import math\n",
        "import glob\n",
        "import io\n",
        "import base64\n",
        "from IPython.display import HTML\n",
        "\n",
        "from IPython import display as ipythondisplay\n",
        "\n",
        "\n",
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
        "  return env\n",
        "\n",
        "env = wrap_env(env)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: gym in /usr/local/lib/python3.6/dist-packages (0.17.3)\n",
            "Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from gym) (1.5.0)\n",
            "Requirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.6/dist-packages (from gym) (1.18.5)\n",
            "Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from gym) (1.3.0)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from gym) (1.4.1)\n",
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
            "0 upgraded, 0 newly installed, 0 to remove and 14 not upgraded.\n",
            "Requirement already satisfied: gym[atari] in /usr/local/lib/python3.6/dist-packages (0.17.3)\n",
            "Requirement already satisfied: numpy>=1.10.4 in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.18.5)\n",
            "Requirement already satisfied: cloudpickle<1.7.0,>=1.2.0 in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.3.0)\n",
            "Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.4.1)\n",
            "Requirement already satisfied: pyglet<=1.5.0,>=1.4.0 in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (1.5.0)\n",
            "Requirement already satisfied: opencv-python; extra == \"atari\" in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (4.1.2.30)\n",
            "Requirement already satisfied: Pillow; extra == \"atari\" in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (7.0.0)\n",
            "Requirement already satisfied: atari-py~=0.2.0; extra == \"atari\" in /usr/local/lib/python3.6/dist-packages (from gym[atari]) (0.2.6)\n",
            "Requirement already satisfied: future in /usr/local/lib/python3.6/dist-packages (from pyglet<=1.5.0,>=1.4.0->gym[atari]) (0.16.0)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.6/dist-packages (from atari-py~=0.2.0; extra == \"atari\"->gym[atari]) (1.15.0)\n",
            "Requirement already satisfied: pyvirtualdisplay in /usr/local/lib/python3.6/dist-packages (1.3.2)\n",
            "Requirement already satisfied: EasyProcess in /usr/local/lib/python3.6/dist-packages (from pyvirtualdisplay) (0.3)\n",
            "Requirement already satisfied: piglet in /usr/local/lib/python3.6/dist-packages (1.0.0)\n",
            "Requirement already satisfied: piglet-templates in /usr/local/lib/python3.6/dist-packages (from piglet) (1.1.0)\n",
            "Requirement already satisfied: attrs in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (20.3.0)\n",
            "Requirement already satisfied: astunparse in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (1.6.3)\n",
            "Requirement already satisfied: markupsafe in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (1.1.1)\n",
            "Requirement already satisfied: Parsley in /usr/local/lib/python3.6/dist-packages (from piglet-templates->piglet) (1.3)\n",
            "Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.6/dist-packages (from astunparse->piglet-templates->piglet) (0.35.1)\n",
            "Requirement already satisfied: six<2.0,>=1.6.1 in /usr/local/lib/python3.6/dist-packages (from astunparse->piglet-templates->piglet) (1.15.0)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sthonRy_HT7K"
      },
      "source": [
        "## Environment\n",
        "\n",
        "\n",
        "From the environment bellow it can be observed that our images have the height and width 210*160. The images are in color, thus, we have three channels.\n",
        "\n",
        "The agent can perform 18 different actions.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ldge3sYcHT7K",
        "outputId": "32824756-b31d-4a33-9b6a-53735af5b758"
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
            "Observation space: Box(0, 255, (210, 160, 3), uint8)\n",
            "Action space: Discrete(18)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "EErbbcqZlAp-"
      },
      "source": [
        "## Preprocessing\n",
        "\n",
        "\n",
        "Preprocess the images is optional but to make them smaller while maintaing the important information makes the training of the neural network easier. \n",
        "\n",
        "The same preprocessing as in the tutorial can be applied because as we can see below the image has an unimportant area in the buttom of the image so we can crop that. The removal of every second row and column still allows to see appearing objects while reducing the size of the image significantly. The tranformation of rgb to grayscale allows to reduces the number of channels from 3 to 1. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HMgk9JJGHT7K",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 520
        },
        "outputId": "61707d7d-692e-4fbe-dde6-2b2fdf67e0d2"
      },
      "source": [
        "obs = env.reset()\n",
        "\n",
        "def preprocess_observation(observation):\n",
        "    img = observation[1:192:2, ::2] #This becomes 96, 80,3\n",
        "    img = img.mean(axis=2) #to grayscale (values between 0 and 255)\n",
        "    img = (img - 128).astype(np.int8) # normalize from -128 to 127\n",
        "    return img.reshape(96, 80, 1)\n",
        "\n",
        "plt.imshow(obs)\n",
        "plt.show()\n",
        "plt.imshow(preprocess_observation(obs).reshape(96,80), cmap='gray', vmin=-128, vmax=127)\n",
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
        "id": "FI624XSHlWQ6"
      },
      "source": [
        "## Model\n",
        "\n",
        "The neural network consist of 3 convolutional layer. The output of the layers gets flatten and then interpreted by a dense hidden layer followed by a dense output layer."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wZMthAygHT7K"
      },
      "source": [
        "def q_network(X_state, name):\n",
        "    prev_layer = X_state / 128.0 # pixels have a range of -128 to 127 so dividing by 128 scales them to a range of [-1.0, 1.0]\n",
        "    initializer = tf.variance_scaling_initializer()\n",
        "    with tf.variable_scope(name) as scope:\n",
        "        prev_layer = tf.layers.conv2d(prev_layer, filters=32, \n",
        "                                      kernel_size=8,strides=4,\n",
        "                                      padding=\"SAME\" ,\n",
        "                                      activation=tf.nn.relu,\n",
        "                                      kernel_initializer=initializer)\n",
        "        prev_layer = tf.layers.conv2d(prev_layer, filters=64,\n",
        "                                      kernel_size=4,strides=2,\n",
        "                                      padding=\"SAME\" ,\n",
        "                                      activation=tf.nn.relu,\n",
        "                                      kernel_initializer=initializer)\n",
        "        prev_layer = tf.layers.conv2d(prev_layer, filters=64,\n",
        "                                      kernel_size=3,strides=1,\n",
        "                                      padding=\"SAME\" , \n",
        "                                      activation=tf.nn.relu,\n",
        "                                      kernel_initializer=initializer)\n",
        "        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1,64 * 12 * 10])\n",
        "        hidden = tf.layers.dense(last_conv_layer_flat,512,\n",
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
        "id": "uLZXLyYJlho-"
      },
      "source": [
        "## Experience"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "qV9KyVZOmJve"
      },
      "source": [
        "Experience replay allows to learn multiple times from an example by extracting a random experiences in a batch. This allows for a better convergence. \n",
        "\n",
        "Instead of using deque becuase according to https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb it allows faster access on random elements and thus speeding up the training process. Additionaly, it employs sampling with replacement wihich reduces the training time again.\n"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uq9r1Xcwg6Il"
      },
      "source": [
        "# code: https://github.com/ageron/handson-ml/blob/master/16_reinforcement_learning.ipynb (partially adapted)\n",
        "class ReplayMemory:\n",
        "    def __init__(self, maxlen):\n",
        "        self.maxlen = maxlen\n",
        "        self.buf = np.empty(shape=maxlen, dtype=np.object)\n",
        "        self.index = 0\n",
        "        self.length = 0\n",
        "        \n",
        "    def append(self, data):\n",
        "        self.buf[self.index] = data\n",
        "        self.length = min(self.length + 1, self.maxlen)\n",
        "        self.index = (self.index + 1) % self.maxlen\n",
        "    \n",
        "    def sample(self, batch_size, with_replacement=True):\n",
        "        if with_replacement:\n",
        "            indices = np.random.randint(self.length, size=batch_size) # faster\n",
        "        else:\n",
        "            indices = np.random.permutation(self.length)[:batch_size]\n",
        "        return self.buf[indices]\n",
        "\n",
        "\n",
        "\n",
        "def sample_memories(batch_size):\n",
        "    cols = [[], [], [], [], []] # state, action, reward, next_state, continue\n",
        "    for memory in agent.replay_memory.sample(batch_size):\n",
        "        for col, value in zip(cols, memory):\n",
        "            col.append(value)\n",
        "    cols = [np.array(col) for col in cols]\n",
        "    return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "gEBbDHgclgX2"
      },
      "source": [
        "## Agent\n",
        "\n",
        "The Q-Learning agent consists of two neural networks (online and target) to improve the stability of the training. \n",
        "\n",
        "The learning rate, momentum, and disount rate could be experimented with to find a better solution.\n",
        "\n",
        "Epsilon:\n",
        "\n",
        "Note, that a number of epsilon values could have been chosen.\n",
        "Epsilon influences whether a random action or an optimal action is selected.\n",
        "With a random action the agent has a higher chance to learn something new.\n",
        "The epsilon here is taken from the tutorial and is in the range of 0.1 and 1\n",
        "Note, that the 0.9 is max-min (1-0.1) and the 2000000 is the number of steps used in the book that inspired the tutorial.\n",
        "Changing it 1000000 would be more aligned with the task's number of steps but it would not make a strong difference.\n",
        "The idea of having a small number multiplied by the number of steps remains the same. \n",
        "Alternatively, one could use a logarithmic decay to make the decay of epsilon prolonged. Allowing the agent to continue learning more over a longer period of time. A reward based decay could be another option.\n",
        "\n",
        "\n",
        "Compared to an agent without experience some slight changes have to be made. First we have to add a ReplayMemory which stores past experiences.\n",
        "Next we have to change the training function. First the new experience is stored in the memory. Next a number of samples according to the batch size (i.e. 25) are fetched from the memory. We get the targets from the target network and train the online network."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "T-ZMcSZAHT7K"
      },
      "source": [
        "# code partially inspired by the lecture: Unit 3.4: Adding Experience Replay to our Agent\n",
        "class QLearningAgent():\n",
        "    def __init__(self, env, learning_rate = 0.001, momentum = 0.95):\n",
        "        self.loss_val = np.infty\n",
        "        self.action_size = env.action_space.n\n",
        "        tf.reset_default_graph()\n",
        "        tf.disable_eager_execution()\n",
        "        self.discount_rate = 0.99\n",
        "        self.checkpoint_path = \"./my_dqn.ckpt\"\n",
        "        self.X_state = tf.placeholder(tf.float32, shape=[None, 96, 80,1])\n",
        "        self.online_q_values, self.online_vars = q_network(self.X_state, name=\"q_networks/online\")\n",
        "        self.target_q_values, self.target_vars = q_network(self.X_state, name=\"q_networks/target\")\n",
        "\n",
        "        #The \"target\" DNN will take the values of the \"online\" DNN\n",
        "        self.copy_ops = [target_var.assign(self.online_vars[var_name]) for var_name, target_var in self.target_vars.items()]\n",
        "        self.copy_online_to_target = tf.group(*self.copy_ops)\n",
        "\n",
        "        \n",
        "        #We create the model for training\n",
        "        with tf.variable_scope(\"train\"):\n",
        "            self.X_action = tf.placeholder(tf.int32, shape=[None])\n",
        "            self.y = tf.placeholder(tf.float32, shape=[None, 1])\n",
        "            self.q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.action_size),axis=1, keepdims=True)\n",
        "            \n",
        "\n",
        "            self.error = tf.abs(self.y - self.q_value)\n",
        "            # clip by value sets any value smaller than 0 to zero and any value larger than 1 to 1\n",
        "            self.clipped_error = tf.clip_by_value(self.error, 0.0, 1.0)\n",
        "            self.linear_error = 2 * (self.error - self.clipped_error)\n",
        "            self.loss = tf.reduce_mean(tf.square(self.clipped_error) + self.linear_error)\n",
        "            \n",
        "            \n",
        "            # declare optimizer \n",
        "            # Instead of simple momentum a number of optimizers could have been explored, e.g. Adam (Nadam, with nesterov momentum), RMSprop, etc.\n",
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
        "        # Memory\n",
        "        replay_memory_size = 250000\n",
        "        self.replay_memory = ReplayMemory(replay_memory_size)\n",
        "\n",
        "\n",
        "    # CHOSSING ACTION \n",
        "    def get_action(self,q_values, step):\n",
        "        epsilon = max(0.1, 1 - (0.9/2000000) * step)\n",
        "        if np.random.rand() < epsilon:\n",
        "            return np.random.randint(self.action_size) # random action\n",
        "        else:\n",
        "            return np.argmax(q_values) # optimal action\n",
        "\n",
        "    # TRAINING\n",
        "    def train(self, experience,batch_size):\n",
        "        #state_val, action_val, reward, next_state_val\n",
        "        # Let's memorize what happened\n",
        "        self.replay_memory.append(experience)\n",
        "\n",
        "        # Sample memories and use the target DQN to produce the target Q-Value\n",
        "        X_state_val, X_action_val, rewards, X_next_state_val, continues = (\n",
        "            sample_memories(batch_size))\n",
        "        \n",
        "        next_q_values = self.target_q_values.eval(\n",
        "            feed_dict={self.X_state: X_next_state_val})\n",
        "        \n",
        "        \n",
        "        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)\n",
        "\n",
        "        y_val = rewards + continues * self.discount_rate * max_next_q_values\n",
        "\n",
        "     \n",
        "        # Train the online DQN\n",
        "\n",
        "        _, self.loss_val = self.sess.run([self.training_op, self.loss], feed_dict={\n",
        "            self.X_state: X_state_val, self.X_action: X_action_val, self.y: y_val})\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "tMWiE7HuxvWI"
      },
      "source": [
        "## Training\n",
        "\n",
        "In order to plot a number of graphs that show as the model's performance a number of lists and counters are created in the beginning. Then the online network decides which action to perform based on the current state. This action is performed. The feedback of the environment is used to train the online network.\n",
        "\n",
        "\n",
        "Because we use experience replay we group the different values the model retrieved by playing the game as on experience sample to the train method.\n",
        "\n",
        "\n",
        "Every 5000 steps the target network is replaced by the online network and every 1000 steps the session is saved."
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r5eu7va0BNZj"
      },
      "source": [
        "**Frame skipping**"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "-3m_0C2fBAmV"
      },
      "source": [
        "In the training process below we use frame skipping by only training the model on every 4th frame. In order for that to work we had to introduce a counter because the previous step was defined by the agents' global_step. Global_step is a tensor variable which increases by 1 when the network sees an example or batch of examples."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GKwRSGtAGRd0"
      },
      "source": [
        "import time\r\n",
        "start_time = time.time()"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MSmxft-SHT7K",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "831547f5-31dd-4f46-e5c9-72f59d88e6d4"
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
        "batch_size=25\n",
        "counter=0\n",
        "\n",
        "with agent.sess:\n",
        "    while True:\n",
        "        step = counter\n",
        "        if step >= n_steps:\n",
        "            break\n",
        "\n",
        "        print(\"\\r\\tTraining step {}/{} ({:.1f})%\\tLoss {:5f}\".format(\n",
        "            step,\n",
        "            n_steps,\n",
        "            step * 100 / n_steps, \n",
        "            agent.loss_val), end=\"\")\n",
        "        \n",
        "\n",
        "        if done: # game over, start again\n",
        "            obs = env.reset()\n",
        "            ep_rewards.append(total_reward)\n",
        "            steps_per_episode.append(steps_counter)\n",
        "            steps_counter=0\n",
        "            total_reward = 0\n",
        "            state = preprocess_observation(obs)\n",
        "\n",
        "        total_perc = int(step * 100 / n_steps)\n",
        "        \n",
        "        # Online DQN evaluates what to do\n",
        "        q_values = agent.online_q_values.eval(feed_dict={agent.X_state: [state]})\n",
        "        action = agent.get_action(q_values, step)\n",
        "        \n",
        "        # Online DQN plays\n",
        "        next_obs, reward, done, info = env.step(action)\n",
        "        next_state = preprocess_observation(next_obs)\n",
        "        if step % 4 == 0:\n",
        "          agent.train((state, action, reward, next_state, 1.0 - done),batch_size=batch_size)\n",
        "        \n",
        "        #env.render()\n",
        "        total_reward+=reward\n",
        "        steps_counter+=1\n",
        "        step_loss.append(agent.loss_val)\n",
        "        step_rewards.append(reward)\n",
        "        state = next_state\n",
        "        counter +=1\n",
        "\n",
        "        # Regularly copy the online DQN to the target DQN\n",
        "        if step % copy_steps == 0:\n",
        "            agent.copy_online_to_target.run()\n",
        "\n",
        "        # And save regularly\n",
        "        if step % save_steps == 0:\n",
        "            agent.saver.save(agent.sess, agent.checkpoint_path)\n"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "\tTraining step 999/1000 (99.9)%\tLoss 0.000431"
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
        "id": "RRXmj5_xGkWr",
        "outputId": "fee73026-6aa6-49be-d862-eec7ca4a4b09"
      },
      "source": [
        "print(\"--- %s seconds ---\" % (time.time() - start_time))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "--- 9.988768339157104 seconds ---\n"
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
        "id": "ewpuX-IA1fGS",
        "outputId": "54b74ba0-942e-4449-eedc-913873a22685"
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
              "1000"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 90
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IukWfCztrD5T",
        "outputId": "5c6aa0cd-586c-490d-d85c-d07a6f1d7d8d"
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
              "1"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 91
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 620
        },
        "id": "Xi45jdZdk3Bp",
        "outputId": "ca495aec-fd22-49ee-ded5-dc62665d2013"
      },
      "source": [
        "plt.plot(range(len(ep_rewards)), ep_rewards)\n",
        "plt.xlabel('Runs')\n",
        "plt.ylabel('Rewards')\n",
        "plt.savefig('images/ep_reward.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "error",
          "ename": "FileNotFoundError",
          "evalue": "ignored",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-92-a0831728c8d7>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mxlabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'Runs'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mylabel\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'Rewards'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msavefig\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m'images/ep_reward.png'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0mplt\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mshow\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/pyplot.py\u001b[0m in \u001b[0;36msavefig\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    721\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0msavefig\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    722\u001b[0m     \u001b[0mfig\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgcf\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 723\u001b[0;31m     \u001b[0mres\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msavefig\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m*\u001b[0m\u001b[0margs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    724\u001b[0m     \u001b[0mfig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdraw_idle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m   \u001b[0;31m# need this if 'transparent=True' to reset colors\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    725\u001b[0m     \u001b[0;32mreturn\u001b[0m \u001b[0mres\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/figure.py\u001b[0m in \u001b[0;36msavefig\u001b[0;34m(self, fname, transparent, **kwargs)\u001b[0m\n\u001b[1;32m   2201\u001b[0m             \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpatch\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mset_visible\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mframeon\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2202\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2203\u001b[0;31m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcanvas\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mprint_figure\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   2204\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2205\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mframeon\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/backend_bases.py\u001b[0m in \u001b[0;36mprint_figure\u001b[0;34m(self, filename, dpi, facecolor, edgecolor, orientation, format, bbox_inches, **kwargs)\u001b[0m\n\u001b[1;32m   2124\u001b[0m                     \u001b[0morientation\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0morientation\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2125\u001b[0m                     \u001b[0mbbox_inches_restore\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0m_bbox_inches_restore\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 2126\u001b[0;31m                     **kwargs)\n\u001b[0m\u001b[1;32m   2127\u001b[0m             \u001b[0;32mfinally\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2128\u001b[0m                 \u001b[0;32mif\u001b[0m \u001b[0mbbox_inches\u001b[0m \u001b[0;32mand\u001b[0m \u001b[0mrestore_bbox\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/backends/backend_agg.py\u001b[0m in \u001b[0;36mprint_png\u001b[0;34m(self, filename_or_obj, metadata, pil_kwargs, *args, **kwargs)\u001b[0m\n\u001b[1;32m    533\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    534\u001b[0m             \u001b[0mrenderer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_renderer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 535\u001b[0;31m             \u001b[0;32mwith\u001b[0m \u001b[0mcbook\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mopen_file_cm\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfilename_or_obj\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m\"wb\"\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mas\u001b[0m \u001b[0mfh\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    536\u001b[0m                 _png.write_png(renderer._renderer, fh, self.figure.dpi,\n\u001b[1;32m    537\u001b[0m                                metadata={**default_metadata, **metadata})\n",
            "\u001b[0;32m/usr/lib/python3.6/contextlib.py\u001b[0m in \u001b[0;36m__enter__\u001b[0;34m(self)\u001b[0m\n\u001b[1;32m     79\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0m__enter__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     80\u001b[0m         \u001b[0;32mtry\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 81\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mnext\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgen\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     82\u001b[0m         \u001b[0;32mexcept\u001b[0m \u001b[0mStopIteration\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     83\u001b[0m             \u001b[0;32mraise\u001b[0m \u001b[0mRuntimeError\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\"generator didn't yield\"\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfrom\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/cbook/__init__.py\u001b[0m in \u001b[0;36mopen_file_cm\u001b[0;34m(path_or_file, mode, encoding)\u001b[0m\n\u001b[1;32m    416\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mopen_file_cm\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath_or_file\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"r\"\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mencoding\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;32mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    417\u001b[0m     \u001b[0;34mr\"\"\"Pass through file objects and context-manage `.PathLike`\\s.\"\"\"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 418\u001b[0;31m     \u001b[0mfh\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mopened\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mto_filehandle\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpath_or_file\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmode\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mencoding\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    419\u001b[0m     \u001b[0;32mif\u001b[0m \u001b[0mopened\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    420\u001b[0m         \u001b[0;32mwith\u001b[0m \u001b[0mfh\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.6/dist-packages/matplotlib/cbook/__init__.py\u001b[0m in \u001b[0;36mto_filehandle\u001b[0;34m(fname, flag, return_opened, encoding)\u001b[0m\n\u001b[1;32m    401\u001b[0m             \u001b[0mfh\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbz2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mBZ2File\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflag\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    402\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 403\u001b[0;31m             \u001b[0mfh\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mopen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mflag\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mencoding\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mencoding\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    404\u001b[0m         \u001b[0mopened\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mTrue\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    405\u001b[0m     \u001b[0;32melif\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mfname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m'seek'\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'images/ep_reward.png'"
          ]
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
        "id": "NSNLzn0pzcgC"
      },
      "source": [
        "plt.plot(range(len(step_loss)), step_loss)\n",
        "plt.xlabel('Runs')\n",
        "plt.ylabel('Loss')\n",
        "plt.savefig('images/step_loss.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "S8t1s718tJsJ"
      },
      "source": [
        "avg=100\n",
        "ep_rewards_avg=[np.mean(ep_rewards[i:i+avg]) for i in range(0,len(ep_rewards),avg)]\n",
        "\n",
        "plt.plot(range(len(ep_rewards_avg)), ep_rewards_avg)\n",
        "plt.xlabel('Runs')\n",
        "plt.ylabel('Rewards')\n",
        "plt.savefig('images/ep_reward_avg_100.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nmBXzF17yKPr"
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
        "plt.show\n",
        "\n"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "tynph6FgiQw8"
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
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "cyn8Qj2viSmb"
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
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wCXUVq4XYF9U"
      },
      "source": [
        "avg=100\n",
        "steps_per_episode_avg=[np.mean(steps_per_episode[i:i+avg]) for i in range(0,len(steps_per_episode),avg)]\n",
        "print(len(step_loss_avg))\n",
        "\n",
        "plt.plot(range(len(steps_per_episode_avg)), steps_per_episode_avg)\n",
        "plt.xticks(range(0,len(steps_per_episode_avg)),range(0,len(steps_per_episode),avg))\n",
        "plt.xlabel('Steps')\n",
        "plt.ylabel('Steps in episode')\n",
        "plt.savefig('images/step_per_episode_100.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "xWXrhIvxHT7K"
      },
      "source": [
        "plt.plot(range(len(ep_rewards)),np.cumsum(ep_rewards) )\n",
        "plt.xlabel('Episodes')\n",
        "plt.ylabel('Total rewards')\n",
        "plt.savefig('images/total_reward.png')\n",
        "plt.show"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}