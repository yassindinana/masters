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
      "name": "Assessment1Task3.ipynb",
      "provenance": [],
      "toc_visible": true
    },
    "accelerator": "GPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "uWiQxheP9P32"
      },
      "source": [
        "#CS5079 Assessment 1 Task 3"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "rUuadx369YGB"
      },
      "source": [
        "This code has been inspired by the tutorial of Unit 3: Reinforcement Learning with OpenAI Gym (CS5079)."
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
        "id": "TjaeH3G-1aFW"
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
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "sthonRy_HT7K"
      },
      "source": [
        "## Environment\n",
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
        "outputId": "7b2fea22-12f7-46ae-b4ed-0d2f87ac7770"
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
        "id": "RDI2I9I_BM8a"
      },
      "source": [
        "## Preprocessing"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "n3Wt1ycaBRBL"
      },
      "source": [
        "Preprocessing the images is optional but to make them smaller while maintaing the important information makes the training of the neural network easier. \n",
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
        "outputId": "3ddb2119-ed8e-419f-92a0-7656c47054f7"
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
        "id": "JPAcnn1JHT7K"
      },
      "source": [
        "## Model\n",
        "\n",
        "According to the paper https://arxiv.org/pdf/1605.01335.pdf the output of the last hidden dense layer in the convolution network is concatenated with the ram input and further processed by dense layers. Thus, the architecture below has 3 conv2d layers, following by 1 flattening, 1 dense layer, 1 concatenation layer, 4 hidden dense layer, and then finally 1 output dense layer. "
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "wZMthAygHT7K"
      },
      "source": [
        "def q_network(X_state,X_state_ram, name):\n",
        "    prev_layer = X_state / 128.0 # pixels have a range of -128 to 127 so dividing by 128 scales them to a range of [-1.0, 1.0]\n",
        "    prev_layer_ram= X_state_ram / 255.0 # ram has a value range of 0 to 255 so dividing it will scale them to [0,1.0]\n",
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
        "        #Ram part\n",
        "        concat= tf.keras.layers.concatenate([hidden,prev_layer_ram],axis=-1)\n",
        "        prev_layer = tf.layers.dense(concat,128,\n",
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
        "id": "oz6GbdfXHT7K"
      },
      "source": [
        "## Agent\n",
        "\n",
        "The Q-Learning agent consists of two neural networks (online and target) to improve the stability of the training. \n",
        "\n",
        "The learning rate, momentum, and discount rate could be experimented with to find a better solution.\n",
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
        "Training\n",
        "\n",
        "In the training function we retrieve the next q value from the target network and use that to compute the target value for the online network. Then we train the online network using the state and action and target."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "T-ZMcSZAHT7K"
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
        "        self.X_state = tf.placeholder(tf.float32, shape=[None, 96, 80,1])\n",
        "        self.X_state_ram = tf.placeholder(tf.float32, shape=[None,128])\n",
        "        self.online_q_values, self.online_vars = q_network(self.X_state,self.X_state_ram, name=\"q_networks/online\")\n",
        "        self.target_q_values, self.target_vars = q_network(self.X_state,self.X_state_ram, name=\"q_networks/target\")\n",
        "\n",
        "        #The \"target\" DNN will take the values of the \"online\" DNN\n",
        "        self.copy_ops = [target_var.assign(self.online_vars[var_name]) for var_name, target_var in self.target_vars.items()]\n",
        "        self.copy_online_to_target = tf.group(*self.copy_ops)\n",
        "\n",
        "        \n",
        "\n",
        "        #We create the model for training\n",
        "        with tf.variable_scope(\"train\"):\n",
        "            self.X_action = tf.placeholder(tf.int32, shape=[None])\n",
        "            self.y = tf.placeholder(tf.float32, shape=[None, 1])\n",
        "            self.q_value = tf.reduce_sum(self.online_q_values * tf.one_hot(self.X_action, self.action_size),axis=1, keepdims=True)\n",
        "             \n",
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
        "    # CHOSSING ACTION\n",
        "    def get_action(self,q_values, step):\n",
        "        epsilon = max(0.1, 1 - (0.9/2000000) * step)\n",
        "        if np.random.rand() < epsilon:\n",
        "            return np.random.randint(self.action_size) # random action\n",
        "        else:\n",
        "            return np.argmax(q_values) # optimal action\n",
        "\n",
        "    # TRAINING\n",
        "    def train(self, state_val, action_val, reward, next_state_val,state_val_ram,next_state_val_ram, continues):\n",
        "        # Compute next_qvalues  \n",
        "        next_q_values = self.target_q_values.eval(feed_dict={self.X_state: np.array([next_state_val]),self.X_state_ram: np.array([next_state_val_ram])})\n",
        "        # Compute best rewards\n",
        "        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)\n",
        "        # Compute target values\n",
        "        y_val = reward + continues * self.discount_rate * max_next_q_values\n",
        "        # Train the online DQN\n",
        "        _, self.loss_val = self.sess.run([self.training_op, self.loss], feed_dict={self.X_state: np.array([state_val]),self.X_state_ram: np.array([state_val_ram]), self.X_action: np.array([action_val]), self.y: y_val})"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ViDJv-CpHT7K"
      },
      "source": [
        "## Training\n",
        "\n",
        "In order to plot a number of graphs that show as the model's performance a number of lists and counters are created in the beginning. Then the online network decides which action to perform based on the current state. This action is performed. The feedback of the environment is used to train the online network.\n",
        "\n",
        "One important change to the screen or the ram models is that we need to have both states. Thus, next to the screen state (and next state) we also have to get the ram state using:\n",
        "\n",
        "state_ram=env.unwrapped._get_ram()\n",
        "\n",
        "\n",
        "Every 5000 steps the target network is replaced by the online network and every 1000 steps the session is saved."
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "MSmxft-SHT7K",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "0568de5e-89b0-4771-ba9c-b2c2d7a9f2bb"
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
        "        \n",
        "        \n",
        "        if done: # game over, start again\n",
        "            obs = env.reset()\n",
        "            ep_rewards.append(total_reward)\n",
        "            steps_per_episode.append(steps_counter)\n",
        "            steps_counter=0\n",
        "            total_reward = 0\n",
        "            state = preprocess_observation(obs)\n",
        "            state_ram=env.unwrapped._get_ram()\n",
        "\n",
        "\n",
        "        total_perc = int(step * 100 / n_steps)\n",
        "        \n",
        "        # Online DQN evaluates what to do\n",
        "        q_values = agent.online_q_values.eval(feed_dict={agent.X_state: [state],agent.X_state_ram: [state_ram]})\n",
        "        action = agent.get_action(q_values, step)\n",
        "        \n",
        "        # Online DQN plays\n",
        "        next_obs, reward, done, info = env.step(action)\n",
        "        next_state = preprocess_observation(next_obs)\n",
        "        next_state_ram=env.unwrapped._get_ram()\n",
        "        agent.train(state, action, reward, next_state,state_ram,next_state_ram, 1.0 - done)\n",
        "        \n",
        "        #env.render()\n",
        "        total_reward+=reward\n",
        "        steps_counter+=1\n",
        "        step_loss.append(agent.loss_val)\n",
        "        step_rewards.append(reward)\n",
        "        state = next_state\n",
        "        state_ram=next_state_ram\n",
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
            "WARNING:tensorflow:From <ipython-input-5-946d54ce9d2b>:10: conv2d (from tensorflow.python.keras.legacy_tf_layers.convolutional) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use `tf.keras.layers.Conv2D` instead.\n",
            "WARNING:tensorflow:From /usr/local/lib/python3.6/dist-packages/tensorflow/python/keras/legacy_tf_layers/convolutional.py:424: Layer.apply (from tensorflow.python.keras.engine.base_layer_v1) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Please use `layer.__call__` method instead.\n",
            "WARNING:tensorflow:From <ipython-input-5-946d54ce9d2b>:24: dense (from tensorflow.python.keras.legacy_tf_layers.core) is deprecated and will be removed in a future version.\n",
            "Instructions for updating:\n",
            "Use keras.layers.Dense instead.\n",
            "\tTraining step 999999/1000000 (100.0)%\tLoss 0.012171"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "ewpuX-IA1fGS",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "e238ed56-79e6-4074-a97a-b37df972efcb"
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
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "IukWfCztrD5T",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "68080d81-f583-420f-bed9-cf0090faac9c"
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
              "1418"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Xi45jdZdk3Bp",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "outputId": "d464e1c4-176c-4534-d660-971bd162c4ee"
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
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<function matplotlib.pyplot.show>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 10
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
        "id": "NSNLzn0pzcgC",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "outputId": "abeaf30f-ebaa-4c3e-8d51-4397e41e7e30"
      },
      "source": [
        "plt.plot(range(len(step_loss)), step_loss)\n",
        "plt.xlabel('Runs')\n",
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
          "execution_count": 11
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
        "id": "S8t1s718tJsJ",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 296
        },
        "outputId": "be54c8d4-d078-4d26-8ba1-5a9bf3def63c"
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
          "execution_count": 12
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEGCAYAAACKB4k+AAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nO3dd3xVVb7+8c83jQ4BQk8gQOgdQrWgiOLYcMaKNEcddQbr6DhF71XnTvOqo446+rOCgKjXsWMDG2MJHUI11BQInUAoIW39/jiHTEQIIck5+5yc5/168SI5OTn7oSRP1t5rr2XOOURERACivA4gIiKhQ6UgIiJlVAoiIlJGpSAiImVUCiIiUibG6wDVkZCQ4JKTk72OISISVhYvXrzLOdfieB8L61JITk5m0aJFXscQEQkrZpZ5oo/p9JGIiJRRKYiISBmVgoiIlFEpiIhIGZWCiIiUUSmIiEgZlYKIiJRRKYiIhJkn5q5j/sbdAXltlYKISBjZtOsgj83NYP6mPQF5fZWCiEgYeXV+JjFRxtWDkwLy+ioFEZEwUVBUwv8tzmFMr9a0bFw3IMdQKYiIhIkP0nPJO1TE+GHtA3YMlYKISJiYkZZJ5xYNGN6pecCOoVIQEQkDK7fsY1l2HuOHdsDMAnYclYKISBiYOT+TurFRXDYoMaDHUSmIiIS4/QVFvLN0K2P7taNJvdiAHkulICIS4t5anMPhohImDOsQ8GOpFEREQphzjhnzs+iX2IQ+iU0CfjyVgohICJu/aQ/rdxwIyigBVAoiIiFtelomTerFcnG/tkE5nkpBRCRE7cgv4JOV27h8UCJ1Y6ODckyVgohIiHpjYTbFpY7xQwN3B/OxVAoiIiGopNTx6vwsTk9JoFOLhkE7rkpBRCQEfb52B1v3FTAhgOscHY9KQUQkBM1Iy6RV4zqM7tEqqMdVKYiIhJjM3QeZt24n44a0JyY6uN+mVQoiIiHm1flZRJlx9eDgnjoClYKISEgpKCrhjUXZnNujFa2bBGYjnYqoFEREQshHK3PZe6goaHcwH0ulICISQmakZdEpoQEjOgduI52KqBRERELE6q37WZy5l2uGticqKnAb6VREpSAiEiJmzM+kTkwUlwd4I52KqBREREJAfkER7yzdwsX92hJfP86zHCoFEZEQ8PbSLRwqLGGiRxeYj1IpiIh4zDnHjLRM+rRrQr+keE+zBKwUzOwlM9thZivLPfawma01s3Qze9vM4st97Pdmtt7MvjezMYHKJSISahZu3kvG9gNBX+foeAI5UpgKnH/MY3OA3s65vkAG8HsAM+sJXA308n/OP80sOIuHi4h4bEZaJo3qxgRtI52KBKwUnHPzgD3HPPapc67Y/24acPQS+1jgNefcEefcJmA9MCRQ2UREQsXO/CN8tDKXywclUj8uxus4nl5TuA74yP92OyC73Mdy/I/9iJndaGaLzGzRzp07AxxRRCSw3liUTVGJY/xQby8wH+VJKZjZvUAxMPNUP9c595xzLtU5l9qiRYuaDyciEiRHN9IZ3qk5KS2Dt5FORYJeCmZ2LXARMN455/wPbwGSyj0t0f+YiEit9VXGDrbkHfZsnaPjCWopmNn5wD3AJc65Q+U+9B5wtZnVMbOOQBdgQTCziYgE2/TvMmnRqA7n9QruRjoVCeSU1FnAd0A3M8sxs+uBp4BGwBwzW2ZmzwI451YBbwCrgY+BKc65kkBlExHxWvaeQ3yZsZNxg5OIDfJGOhUJ2KVu59y44zz8YgXP/zPw50DlEREJJa8uyMKAq4d4f29CeaFTTyIiEeJIcQlvLMxmdI9WtI2v53WcH1ApiIgE2ccrt7H7YGFIXWA+SqUgIhJkM9Iy6dC8PqenJHgd5UdUCiIiQbR2234Wbt7LeA830qmISkFEJIhmpGUSFxPFFYOSTv5kD6gURESC5MCRYt5esoWL+rahaQPvNtKpiEpBRCRI3lm6hYOFJSF5gfkolYKISBAc3UinZ5vGDPB4I52KqBRERIJgceZe1m7LZ+LwDpiF3gXmo1QKIiJBMCMtk0Z1Yhjb3/uNdCqiUhARCbDdB47w4Ypt/Gxgu5DYSKciKgURkQD7v8U5FJaUMj6ELzAfpVIQEQmg0lLHzPmZDO3YjK6tGnkd56RUCiIiAfTVup1k7wmtjXQqolIQEQmgmWmZJDSsw5herb2OUikqBRGRAMnZe4jP1+7gqsGJxMWEx7fb8EgpIhKGZi3IAmBciG2kUxGVgohIABQUlfDagmxGdW9FYtP6XsepNJWCiEgAzE7PZffBQq4dkex1lFOiUhARqWHOOaZ+u5mUlg05LaW513FOiUpBRKSGLcnKY8WWfUwekRzS6xwdj0pBRKSGTf12M43qxvCzAe28jnLKVAoiIjVo+/4CPlqRy5WpSTSoE9rrHB2PSkFEpAbNTMukxDkmDQ+PO5iPpVIQEakhR4pLeHVBFud0b0mH5g28jlMlKgURkRoyOz2XXQcKmRxm01DLUymIiNSA8tNQT09J8DpOlakURERqwNLsPNJz9jE5xLfbPBmVgohIDZj6zWYa1YnhZwMTvY5SLSoFEZFq2r6/gA9X5HJFmE5DLU+lICJSTTPnZ4X1NNTyVAoiItVwpLiEV+dnMqpbS5ITwnMaankqBRGRavhwRfhPQy1PpSAiUkXOOV7+ZjOdWzTgjC7hOw21PJWCiEgVlU1DDcPVUE8kYKVgZi+Z2Q4zW1nusWZmNsfM1vl/b+p/3MzsH2a23szSzWxgoHKJiNSUad/Wjmmo5QVypDAVOP+Yx34HfOac6wJ85n8f4CdAF/+vG4FnAphLRKTaduwvYHZ6LpenJtIwzKehlhewUnDOzQP2HPPwWGCa/+1pwKXlHn/F+aQB8WbWJlDZRESq6+g01MnDk72OUqOCfU2hlXMu1//2NqCV/+12QHa55+X4HxMRCTmFxaXMnJ/F2bVkGmp5nl1ods45wJ3q55nZjWa2yMwW7dy5MwDJREQq5puGeqTWTEMtL9ilsP3oaSH/7zv8j28Bkso9L9H/2I84555zzqU651JbtGgR0LAiIsfz8reb6dSiAWeE8WqoJxLsUngPmOx/ezLwbrnHJ/lnIQ0D9pU7zSQiEjKWZu1leXYek4cnExVVO6ahlhewS+ZmNgs4C0gwsxzgfuBvwBtmdj2QCVzpf/qHwAXAeuAQ8PNA5RIRqY5p326mYZ0YLhtUe6ahlhewUnDOjTvBh845znMdMCVQWUREasKO/AJmr8hl/NAOtWoaanm6o1lEpJJenZ9FUYmrlReYj6pUKZjZ7WbW2H/O/0UzW2Jm5wU6nIhIqPjPNNQWdKxl01DLq+xI4Trn3H7gPKApMBHf9QERkYjw0cpcdubXzmmo5VW2FI5eYr8AmO6cW1XuMRGRWu/lbzbTKaEBZ3ap3VPhK1sKi83sU3yl8ImZNQJKAxdLRCR0LMvOY1l2HpOGd6iV01DLq+zl8+uB/sBG59whM2uOpo2KSISo7dNQy6uwFI6zhHWn2rJmuIhIZezIL+CD9K2MH9qBRnVjvY4TcCcbKTzq/70uMAhIx3ctoS+wCBgeuGgiIt6bNT+bohLHpOEdvI4SFBVeU3DOne2cOxvIBQb51xwaBAzgBGsTiYjUFoXFpcyYn8lZ3VrQqUVDr+MERWUvNHdzzq04+o5zbiXQIzCRRERCQ6RMQy2vsheaV5jZC8AM//vj8Z1KEhGptaZ+u5mOCQ0YWcunoZZX2ZHCtcAq4Hb/r9Vo9pGI1GLLs/NYmhUZ01DLO+lIwcyigY/81xYeC3wkERHvTft2Mw3iork8AqahlnfSkYJzrgQoNbMmQcgjIuK5nflHeD99K1ekJkXENNTyKntN4QC+6wpzgINHH3TO3RaQVCIiHpq1ICuipqGWV9lSeMv/S0SkVissLmVGWiYju0bONNTyKlUKzrlpgQ4iIhIKPl61jR35R3josmSvo3iiUqVgZl2AvwI98d3dDIBzrlOAcomIeGLqN5tIbl6fkV0jZxpqeZWdkvoy8AxQDJwNvMJ/7lkQEakV0nPyWJKVx6ThyRE1DbW8ypZCPefcZ4A55zKdcw8AFwYulohI8E09Og01NbKmoZZX2QvNR8wsClhnZrfgW/co8q7AiEittevAET5Ynsu4IUk0jrBpqOVVdqRwO1AfuA3faqkTgMmBCiUiEmyz5mdRWFLKpAha5+h4KjtS2OOcO4DvfgUtbyEitUpRiW811DO7tqBzBE5DLa+yI4WXzGyDmb1mZlPMrE9AU4mIBIlzjme/3MD2/Ue4dkTk3ax2rMrepzDSzOKAwcBZwGwza+icaxbIcCIigbS/oIjfvpnORyu3MaZXK87q2tLrSJ6r7H0KpwNn+H/FAx8A/w5gLhGRgFq1dR9TZi4he+9h7r2gBzec0RFtN1z5awpfAovx3cD2oXOuMGCJREQCyDnHawuzuf+9VTSrH8frNw4jNVknPY6qbCkkAKcBZwK3mVkp8J1z7r8ClkxEpIYdPFLMfe+s5O2lWzijSwKPX9Wf5g3reB0rpFT2mkKemW0EkoBEYAQQuRN5RSTsrNuezy9nLmHjzgPcdW5XppydErF3LVekstcUNgJrga/xLXfxc51CEpFw8daSHO59eyUN6sQw4/qhjEhJ8DpSyKrs6aMU51xpQJOIiNSwgqISHnhvFa8tzGZox2Y8OW4ALRvXPfknRrBKl4KZPQO0cs71NrO+wCXOuT8FMJuISJVt2nWQX85YzNpt+Uw5uzN3ju5KTHRlb82KXJX9G3oe+D1QBOCcSweuDlQoEZHqmJ2ey8VPfs22/QW8/PPB/GZMdxVCJVV2pFDfObfgmDm8xQHIIyJSZUeKS/jL7DVM+y6TAe3jefqagbSNr+d1rLBS2VLYZWadAQdgZpcDuQFLJSJyirL3HOKWV5ewPGcfN5zekXvO705cjEYHp6qypTAFeA7obmZbgE3A+IClEhE5BXNWb+euN5bhgGcnDOL83q29jhS2KnufwkZgtJk1wHcd4hC+awqZVTmomd0J3IBv5LEC38qrbYDXgOb47p6eqGmvIlKRopJSHvnke/7fvI30bteYf14ziPbN63sdK6xVOLYys8Zm9nsze8rMzsVXBpOB9cCVVTmgmbXDty9DqnOuNxCNr2AeAh5zzqUAe4Hrq/L6IhIZcvcdZtxzafy/eRuZMKw9b948QoVQA052wm060A3fT/O/AL4ArgB+6pwbW43jxgD1zCwG3+Y9ucAo4E3/x6cBl1bj9Su0OHMPV/2/79hfUBSoQ4hIAM3L2MmF//ia1bn7eeLq/vzp0j7UjY32OlatcLLTR52cc30AzOwFfN+82zvnCqp6QOfcFjN7BMgCDgOf4jtdlOecOzqjKQdod7zPN7MbgRsB2rdvX6UMsdFRLNi8hyc/W8e9F/as0muISPCVljoen5vBk1+sp2vLRjw9fiApLSN7U5yadrKRQtmP0s65EiCnOoUAYGZNgbFAR6At0AA4v7Kf75x7zjmX6pxLbdGiRZUy9E2M56rUJF7+ZjPrd+RX6TVEJLicc/zxg9X84/P1XDYwkXemnKZCCICTlUI/M9vv/5UP9D36tpntr+IxRwObnHM7nXNFwFv4VmCN959OAt+ie1uq+PqVcveYbtSLi+bB91fjnAvkoUSkBjz1+XqmfruZ607ryMOX96VenE4XBUKFpeCci3bONfb/auSciyn3duMqHjMLGGZm9c13N9w5wGp81ysu9z9nMvBuFV+/UhIa1uHX53bl3+t28enq7YE8lIhU04y0TB6dk8HPBrTjvgt7aDOcAAr6nR3Oufn4LigvwXcBOwrfPRC/BX5tZuvxTUt9MdBZJgzrQNdWDfmfD1ZTUFQS6MOJSBV8kL6V/3p3JaO6t+Shy/tquesA8+R2P+fc/c657s653s65ic65I865jc65Ic65FOfcFc65I4HOERsdxQMX9yJn72Gem7cx0IcTkVM0L2Mnd76+jNQOTXn6moHEav2igIv4v+ERKQlc0Kc1//xyPTl7D3kdR0T8lmbt5eYZi+ncoiEvTB6sawhBEvGlAPCHC3oA8NcP13qcRETAt0vaz6cuJKFhHV65fghN6mmjx2BRKQCJTevzy5EpzF6Ry7frd3kdRySi5ew9xMQXFxAbHcWM64fSspE2xQkmlYLfTSM7kdi0Hg+8v4riEm0yJ+KF3QeOMOnFBRwsLOaV64Zo2QoPqBT86sZGc9+FPcnYfoDpaVVa509EqiG/oIhrX17IlrzDvHTtYHq0qeqsd6kOlUI5Y3q14owuCfx9Tga7DwR88pOI+BUUlXDjK4tZnbufZyYMZHByM68jRSyVQjlmxv0X9+RwYQkPf/K913FEIkJxSSm3v7aU7zbu5pEr+jKqeyuvI0U0lcIxUlo24toRyby+KJv0nDyv44jUas457n17JZ+s2s5/X9STnw5I9DpSxFMpHMdto7vQvEEd7n9vFaWlWhdJJFAe+vh7Xl+Uza2jUrju9I5exxFUCsfVuG4svz2/G0uz8nh7aUDX5ROJWM/N28CzX21g/ND2/Prcrl7HET+VwglcNjCR/knx/PWjteRrMx6RGvXGomz+8uFaLuzbhj+O7a0F7kKISuEEoqKMBy/pxa4DR3jy8/VexxGpNT5ZtY3f/SvdN9Pvyn5Ea4G7kKJSqEC/pHiuTE3kpa83sX7HAa/jiIS97zbs5tZZS+mTGM+zEwZRJ0brGYUalcJJ3HN+d+rFRvPg+6u0GY9INazcso9fvLKI9s3qM/XawTSoc7LdgMULKoWTSGhYhzv8m/HM0WY8IlWycecBJr+0gCb1Ypl+/RCaNojzOpKcgEqhEiYN70CXlg35n9najEfkVG3bV8DEFxfggFeuH0KbJvW8jiQVUClUQmx0FA9c0ovsPYd5XpvxiFRa3qFCJr00n7xDhUz9+WA6t2jodSQ5CZVCJZ2WksBPerfm6S/XszXvsNdxRELeocJifj51IZt3HeL5San0TYz3OpJUgkrhFNx7YQ+cgz9/uMbrKCIhzTnHbbOWsjw7j3+M68+IlASvI0klqRROQWLT+vzyrM7MTs/luw27vY4jIco5x9zV2yP6+tPX63cxd80OfveT7pzfu43XceQUqBRO0c0jO9Muvh4PajMeOYGvMnZywyuL+N+PI3OlXeccj83JoG2Tukwekex1HDlFKoVTVDc2mv+6qAdrt+Uzc36W13EkBM3wb9L0ynebWb8j39swHpi3bhdLsvL41dkpujktDKkUqmBMr9acnpLAo59+r8145Ady9h7i87U7GDckifpx0Tz4/uqIuunROcfjc32jhCtTk7yOI1WgUqiCo5vxHCos4ZFPI/MUgRzfrAW+0eOUs1O4Y7Tvpse5a3Z4nCp4vsrYydKsPKaMSiEuRt9ewpH+1aqoS6tGTB6RzGsLs1mRsy+gxyopdRH102a4OlJcwusLsxnVvRWJTeszcXgHUlo25E+zV3OkuPZfdHbO8djcdbSLr8cVgzRKCFcqhWq4fXQXmjeI4/73VtboZjwFRSV8t2E3j8/NYNxzafT8748Z8/g8Pl21TeUQwj5euY1dBwqZOLwD4Lvp8b8v6knm7kO89PVmb8MFwZff72R5dh63aJQQ1rQiVTU0rhvLPed3554303ln2RZ+NrBqWwkeKixmceZe5m/cw/xNu1mevY/CklLMoGebxlw9OIl/r9/FjdMXM7B9PL89vztDOzWv4T+NVNeMtEw6NK/PGeXm5J/ZtQWje7Tiqc/XcdnAdrRsXNfDhIHjGyVkkNi0HpcP0paa4UylUE2XD0xk5vws/vrRWs7t2YpGdWNP+jn5BUUsKlcCK3L2UVzqiI4yerdrwrWnJTO0YzNSk5vRpJ7v9YpLSnlzcQ6Pz13HVc+lcVa3FvxmTDd6tW0S6D+iVMKa3P0s3LyXP1zQnahj9ge478IenPfYPB76+HsevbKfRwkD64vvd5Ces4+HLutDbLRGCeFMpVBNRzfjufTpb3jq8/X8/oIeP3rOvsNFLNzkK4D5m/awcss+Sh3ERht9E+O58cxODO3UnEEdmtLwBMsJx0RHcfWQ9lw6oB3Tvt3MP7/cwIX/+Jqx/dvy63O70qF5g0D/UaUCM9IyiYuJOu659OSEBlx3ekee/WoDE4a1Z0D7ph4kDBzfjKN1JDWrV+XRsoQOlUIN6J8UzxWDEnnpm01cOTiJpvXjWOAvgPkb97Bm236cg7iYKPonxXPL2SkM7dScge2bUi/u1OZx142N5qaRnbl6SHuem7eBF7/exOz0XK4Z2p5bRqXQslHtPD0RyvILinhn6RYu7tv2hEtC3zIqhX8tyeGB91fz9i9H/Gg0Ec4+W+MbJfzvZX01SqgFLJwvXKamprpFixZ5HQOAnflHGPXIlwDkHykGoG5sFAPbN2Vox+YM7dSM/knx1I2t2Zt5duwv4B+fr+O1BdnERkdx/ekduXFkJxpX4jSW1Izp323mv95dxTtTTqN/0okXfXtzcQ53/99yHr2iH5fVkvPuzjkufupr9h8u5rO7RqoUwoSZLXbOpR73YyqFmvPusi28u2wrgzo0ZWjHZvRNjA/aLIzNuw7y6JwM3l++lfj6sUw5K4WJwzvUeAnJDznnGPP4POrERPPeLadVuAF9aanjp898y9a8w3xx91knPFUYTuas3s4vXlnEw5f35QrdrBY2KioF1XoNGtu/HS9dO5gpZ6eQmtwsqNPykhMa8OS4AXxw6+n0TYznzx+u4exHvuSNhdlaoymAFmzaQ8b2A0wc1qHCQgDf9acHLu7JzvwjPPX5+iAlDJyjdy93aF6fnw5o53UcqSEqhVqmd7smvHLdEF79xVBaNq7LPf9KZ8zj8/h4Za7ucQiA6WmZNK4bw8X92lbq+QPaN+WygYm89PUmNu86GOB0gfXp6u2s2rqfW0d1IUanjWoN/UvWUiM6J/DOr0bw7IRBANw8YwmX/vNbvt2wy+NktceO/AI+XrmNywclndKEgd+e343YaONPs8N3X47SUt+Mo44JDbi0f+UKUcKDJ6VgZvFm9qaZrTWzNWY23MyamdkcM1vn/712zdvzgJlxfu/WfHLHmfzvZX3Zsb+Aa56fz6SXFrByS2CX5ogEry/IprjUMX5Y+1P6vJaN63LLqC7MXbOdeRk7A5QusD5dvY01ufu5dVSKRgm1jFf/mk8AHzvnugP9gDXA74DPnHNdgM/870sNiImO4srBSXxx91nce0EP0nPyuOjJr7nl1SXs0iqvVVJcUsqsBVmcnpJQpX2Hrzs9mQ7N6/PHD1ZTFGbXfI6OEjolNOCSSp42k/AR9FIwsybAmcCLAM65QudcHjAWmOZ/2jTg0mBnq+3qxkbzizM7Me+es7l1VErZzJFI3iGsqj5fu4Ot+wqYMKxDlT6/Tkw0913Yk/U7DjD9u8waThdYn6zaxtpt+dx2jq4l1EZe/It2BHYCL5vZUjN7wcwaAK2cc7n+52wDWnmQLSI0rhvLXed147Gr+rM0K48/vL1CF6FP0fS0TFo3rsvoHi2r/Bqje7TkjC4JPDY3I2z25SgbJbRoUOmL6xJevCiFGGAg8IxzbgBwkGNOFTnfd6jjfpcysxvNbJGZLdq5MzzPx4aKC/q04c7RXXlryRaem7fR6zhhY9Oug/x73S6uGdq+Wj8p/3BfjowaTBg4H63cxvfb87n9nC5E16K7suU/vCiFHCDHOTff//6b+Epiu5m1AfD/ftydSZxzzznnUp1zqS1atAhK4NrstnNSuLBPG/728Vo+X7vd6zhhYWZaJjFRxtWDq3+zVkrLRkwa3oHXFmaxamtoX/wvLXU88VkGKS0bclFfjRJqq6CXgnNuG5BtZt38D50DrAbeAyb7H5sMvBvsbJHIzHjkin70atuY22YtI2N75O0pfCoOF5bwf4tzGNOrdY0tg33H6K40rR/Hg++F9tadH67MJWP7AW7TKKFW8+oq0a3ATDNLB/oDfwH+BpxrZuuA0f73JQjqxUXz/KRU6sVFc8O0Rew9WOh1pJD1fvpW9h0uqvIF5uNpUi+Wu8/rxoLNe/ggPffkn+CBklLHE3PX0aVlQy7s08brOBJAnpSCc26Z/xRQX+fcpc65vc653c65c5xzXZxzo51ze7zIFqnaNKnHcxMHsW1/Ab+cuTjspkkGy8y0TFJaNmRYp2Y1+rpXDU6iZ5vG/PXDNRwuDL3ZYLNX5LJuh0YJkUDzyaTMgPZNeeiyPqRt3MP9760K6VMZXliencfynH2VWufoVEVHGQ9c0out+wp49qsNNfra1eUbJWTQtZVGCZFApSA/8NMBidw8sjOvzs9ielp4zZ8PtBlpmdSPi+anAwOz+NuQjs24qG8bnv1qAzl7DwXkGFXxQfpWNuw8yO3ndK1V+0DI8akU5Ed+M6Ybo3u05MH3V/PNeq2VBJB3qJD3lm/l0gHtArpXxe8v6IEZ/PWjtQE7xqkoKXX847N1dG/diJ/0bu11HAkClYL8SHSU8fjVA+jcogG/mrmETWG+mmdNeHNxDkeKS5kwtOYuMB9Pu/h63DyyM7PTc0nbuDugx6qM95cfHSV00SghQqgU5Lga1onhhUmDiTK4ftpC9h0u8jqSZ0pLHTPSMhnUoSk92zYO+PFuOrMz7eLr8eD7qykp9e66TnFJadkoYUwvjRIihUpBTqh98/o8M2EQWbsPcduspZ5+gyopdZ6t0fTNhl1s3n2IiTU4DbUi9eKi+cMFPViTu59ZC7KCcszjeW/5VjbuOsgdozVKiCQqBanQsE7N+ePY3nyVsZO/fujN+v/zMnZy7t+/4uxHvvRkY5rp32XSrEEcP+kTvJ+WL+jTmqEdm/Hop9+z71DwR2nFJaU8+fl6erRpzHk9NUqIJCoFOalrhrbn2hHJvPD1Jt5YmB2042bvOcRN0xcx6aUFlDrfSOGa59PI3hO8mTlb8w4zd812rhqcRJ2Y4O137VsXqRf7Dhfx2Nzgr4v07rKtbNIoISKpFKRS7ruwB6enJHDvOytYuDmw9xUWFJXwxNx1jP77V8zL2MVvxnTjkzvPZMYNQzlYWMK459PYmnc4oBmOmrUgCwdcM+TUNtKpCT3bNmbckPZMT8sM6vIjvlHCOnq2acx5PRfn9mYAAAzYSURBVLVYcaRRKUilxERH8fQ1A0lsWp+bpy8OyDx65xxzVm/n3Me+4rG5GYzu2YrP7hrJlLNTqBMTTa+2TZhx/VD2HS5i3PNpbNtXUOMZyissLmXWgmxGdWtJUrP6AT3Widx1XjcaxEXzPx8Eb12kt5duYfPuQ9wxukuN36QnoU+lIJXWpH4sz09KpbCklBumLeLgkeIae+1Nuw5y3dSF/OKVRdSNiebVG4by9DUDaRtf7wfP65PYhGnXDWFX/hGueSGNHfmBK4ZPVm1j14EjTBgenAvMx9OsQRx3ntuVf6/bxZzVgV/FtriklKe+WE/vdo05V6OEiKRSkFOS0rIhT10zkIzt+dz5+jJKqzkj6VBhMQ9/spYxj81j4ea93HdhDz68/QxGpCSc8HMGtm/K1OuGkJtXwIQX5gdsg5rpaZkkNavHyC7eLtE+YVgHurRsyJ9mrwn4DKy3lm4hc/ch7jinq0YJEUqlIKdsZNcW3HthTz5dvb3KF0Gdc8xOz+WcR7/i6S82cFHfNnx+10huOKMTsZXYuGZwcjNeunYwWXsOMf6F+TW+smvG9nwWbNrD+KEdPL/QGhsdxf0X9yJrzyFe+mZTwI5T5L+W0DexCedUY0c5CW8xXgeQ8HTdaclkbMvnyc/Xk9KyIWP7V349oHXb87n/vVV8u2E3Pdo05h/jBjA4+dRXHR3euTnPT0rl+mmLmPjSfGbeMIwm9WpmCYoZaZnExURxZWr1N9KpCad3SeDcnq14bE4GX6zdQZ928fRLakKfdk1Ibt6gRorrrSU5ZO85zIOX9NIoIYJZOK+EmZqa6hYtWuR1jIhVWFzKhBfmszwnjzduGk6/pPgKn59fUMQTc9cx9dvN1I+L5jdjunHN0A7VXor5i+93cNMri+nRtjEzrh9Co2quTXTgSDHD/vIZ5/Vsxd+v6l+t16pJuw4c4Z9fbGB5Th6rtu6joMi3vHmjujH0adeEvonx9E1sQt/EJrSLr3dK39gLi0sZ9eiXNG8QxztTTlMp1HJmttg5l3rcj6kUpDp2HzjCJU99Q3FpKe/dcjqtjrMbmXOOd5Zt4S8frmXXgSNcPTiJu8/rRvOGdWosx5zV2/nljMX0S4rnleuG0KBO1QfBM9Iyue+dlbz1qxEMbN+0xjLWpOKSUtbtOEB6Th7pOftIz9nH2m37KSrxfT03bxBHn8Qm9C1XFhXtFPfagix+99YKXr52MGd316mj2k6lIAG1Jnc/lz3zLV1aNuT1m4ZTN/Y/N3mt2rqPB95bxcLNe+mXFM8fL+l10hFFVX20IpdbZi1lUIemTP35YOrHnXoxOOf4yRP/JjrK+ODW08PqJ+YjxSWszc0nfcs+VvjLImN7PkfnArRuXJc+iU3ol9iEPonx9G3XhKYN4igsLuXsR76kRaM6vP2rEWH1Z5aqqagUdE1Bqq1Hm8Y8flV/bpqxmHveTOeJq/uz/3Axj875nhlpmcTXj+Ohy/pwxaCkgF60/UmfNvy9pJQ7X1/GL15ZxIuTB/+goCpjUeZe1m7L528/6xN23xzrxETTLyneX7q+abSHC0tYtXWffzSRR/qWfT+Y2prUrB6tG9dlS95h/vzT3mH3Z5aap1KQGnFer9bcfV43Hv7ke0qd49sNu8k7VMjEYR349bndaFI/cHsQlDe2fzuKSxx3v7mcm6Yv5rlJg05peYrp32XSqG4Ml/RvG8CUwVMvLprU5GaklruQv7+giJVb9rHCf9ppeU4eo7q3ZGRXb6feSmhQKUiN+dVZncnYns+7y7YyOLkpD14yNChLTR/rskGJFJeW8tt/rWDKzCX8c/wg4mJOPs1114EjfLQyl/FDO1Tp1FO4aFw3lhGdExjR+cT3gkjkqr3/8yXozIyHL+/H5BHJDEiK9/RUxFWD21NU4rjvnZXcOmsJT10z8KT3P7y+MJuiEseEIC2RLRKKdPOa1Ki4mCgGtm8aEuemJwzrwP0X9+STVdu58/VlFJeUnvC5JaWOV+dnMaJzc1JaNgxiSpHQopGC1Go/P60jRSWl/OXDtcRGR/HIFf2Oe1/EF2t3sCXvMPdd2MODlCKhQ6Ugtd6NZ3amqMTx8CffExNlPHRZ3x/NgpqelkmrxnUYrUXgJMKpFCQiTDk7hcLiUp74bB2xMVH8+dL/TL/M3H2QrzJ2csfoLpVad0mkNlMpSMS4Y3QXikpK+eeXG4iLjuL+i3tiZsycn0V0lHH14OBvpCMSalQKEjHMjN+M6UZRSSnP/3sTMVHG3WO68caibM7r2YrWTU68DIRIpFApSEQxM/5wQQ+KShwvfL2Jpdl55B0qYqKmoYoAKgWJQGbG/Rf3pLCklFfnZ9GpRQOGd27udSyRkKBSkIhkZvxpbG/aN6tPf49vtBMJJSoFiVhRUcbNIzt7HUMkpGj+nYiIlFEpiIhIGZWCiIiUUSmIiEgZlYKIiJRRKYiISBmVgoiIlFEpiIhIGXPOeZ2hysxsJ5BZxU9PAHbVYJxAC6e84ZQVwitvOGWF8MobTlmhenk7OOdaHO8DYV0K1WFmi5xzqV7nqKxwyhtOWSG88oZTVgivvOGUFQKXV6ePRESkjEpBRETKRHIpPOd1gFMUTnnDKSuEV95wygrhlTecskKA8kbsNQUREfmxSB4piIjIMVQKIiJSJiJLwczON7PvzWy9mf3O6zwnYmZJZvaFma02s1VmdrvXmSrDzKLNbKmZfeB1loqYWbyZvWlma81sjZkN9zpTRczsTv//g5VmNsvM6nqdqTwze8nMdpjZynKPNTOzOWa2zv97Uy8zHnWCrA/7/y+km9nbZhbvZcbyjpe33MfuMjNnZgk1cayIKwUziwaeBn4C9ATGmVlPb1OdUDFwl3OuJzAMmBLCWcu7HVjjdYhKeAL42DnXHehHCGc2s3bAbUCqc643EA1c7W2qH5kKnH/MY78DPnPOdQE+878fCqby46xzgN7Oub5ABvD7YIeqwFR+nBczSwLOA7Jq6kARVwrAEGC9c26jc64QeA0Y63Gm43LO5Trnlvjfzsf3Taudt6kqZmaJwIXAC15nqYiZNQHOBF4EcM4VOufyvE11UjFAPTOLAeoDWz3O8wPOuXnAnmMeHgtM8789Dbg0qKFO4HhZnXOfOueK/e+mAYlBD3YCJ/i7BXgMuAeosRlDkVgK7YDscu/nEOLfaAHMLBkYAMz3NslJPY7vP2mp10FOoiOwE3jZf6rrBTNr4HWoE3HObQEewfcTYS6wzzn3qbepKqWVcy7X//Y2oJWXYU7BdcBHXoeoiJmNBbY455bX5OtGYimEHTNrCPwLuMM5t9/rPCdiZhcBO5xzi73OUgkxwEDgGefcAOAgoXNq40f85+LH4iuztkADM5vgbapT43zz30N+DryZ3Yvv1O1Mr7OciJnVB/4A/HdNv3YklsIWIKnc+4n+x0KSmcXiK4SZzrm3vM5zEqcBl5jZZnyn5UaZ2QxvI51QDpDjnDs68noTX0mEqtHAJufcTudcEfAWMMLjTJWx3czaAPh/3+FxngqZ2bXARcB4F9o3cXXG9wPCcv/XWyKwxMxaV/eFI7EUFgJdzKyjmcXhu1j3nseZjsvMDN857zXOub97nedknHO/d84lOueS8f29fu6cC8mfZp1z24BsM+vmf+gcYLWHkU4mCxhmZvX9/y/OIYQvjJfzHjDZ//Zk4F0Ps1TIzM7Hd+rzEufcIa/zVMQ5t8I519I5l+z/essBBvr/X1dLxJWC/0LSLcAn+L6o3nDOrfI21QmdBkzE9xP3Mv+vC7wOVYvcCsw0s3SgP/AXj/OckH9E8yawBFiB72s3pJZlMLNZwHdANzPLMbPrgb8B55rZOnyjnb95mfGoE2R9CmgEzPF/rT3rachyTpA3MMcK7RGSiIgEU8SNFERE5MRUCiIiUkalICIiZVQKIiJSRqUgIiJlYrwOIBIuzKwE33TQGGATMDEM1ksSOSUaKYhU3mHnXH//KqV7gCleBxKpaSoFkar5Dv9Cimb2pZml+t9O8C87gJlda2ZvmdnH/v0E/tf/eLSZTfXvi7DCzO706g8hciydPhI5Rf49Oc7Bv+z2SfTHt7rtEeB7M3sSaAm08484CKXNXEQ0UhCpvHpmtoz/LAE9pxKf85lzbp9zrgDf2kodgI1AJzN70r/eTsiufCuRR6UgUnmHnXP98X1jN/5zTaGY/3wtHbtF5pFyb5cAMc65vfh2evsSuJkQ35BIIotKQeQU+VfQvA24y78L2mZgkP/Dl5/s8/176UY55/4F3EdoL9ktEUbXFESqwDm31L+66jh8O6K9YWY3ArMr8ent8O34dvSHslDaC1ginFZJFRGRMjp9JCIiZVQKIiJSRqUgIiJlVAoiIlJGpSAiImVUCiIiUkalICIiZf4/r9zebQEaSSQAAAAASUVORK5CYII=\n",
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
        "id": "nmBXzF17yKPr",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "outputId": "081b1969-5ece-40fe-d332-2428a52a742b"
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
          "execution_count": 13
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
        "id": "wCXUVq4XYF9U",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 313
        },
        "outputId": "50184ddc-2f7c-4674-9563-f78e8bf7911f"
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
          "execution_count": 14
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
        "id": "HRrvdSPWmNmE",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 300
        },
        "outputId": "188c613d-4fee-4e67-98d3-69742f1ebe01"
      },
      "source": [
        "plt.plot(range(len(ep_rewards)),np.cumsum(ep_rewards) )\n",
        "plt.xlabel('Episodes')\n",
        "plt.ylabel('Total rewards')\n",
        "plt.savefig('images/total_reward.png')\n",
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
          "execution_count": 15
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
        "id": "FB4eq27g84e-",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 286
        },
        "outputId": "60d3f5e4-9fa7-467a-93b3-30d2bb2416c1"
      },
      "source": [
        "plt.imshow(env.render('rgb_array'))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "<matplotlib.image.AxesImage at 0x7f33053c3ba8>"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 16
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