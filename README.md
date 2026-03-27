# Projet SO101 

S'aider de la documentation Hugging Face LeRobot : https://huggingface.co/docs/lerobot/

## Commandes de base

### Gestion UV

Pour installer UV :

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Synchroniser les dépendances :

```
uv sync
```

### Gestion des bras + caméras

Pour détecter les ports des bras :

```
uv run lerobot-find-port
```

Cela permet de voir quel bras (Leader ou Follower) est connecté à quel port (ex : /dev/ttyACM0 pour Leader et /dev/ttyACM1 pour Follower).

On donne les droits sur les ports :

```
chmod 666 /dev/ttyACM0
chmod 666 /dev/ttyACM1
```

Pour détecter les caméras :

```
uv run lerobot-find-cameras
```

Cela permet de voir quel caméra (wrist ou overhead) est connecté à quel chemin (ex : /dev/video2 pour wrist et /dev/video4 pour overhead) grâce aux photos prises et sauvegardées dans le dossier `outputs/captured_images/`.

On peut alors définir ces cinq variables qui seront utilisées dans les commandes qui suivent :

```
LEADER_PORT="/dev/ttyACM0"
FOLLOWER_PORT="/dev/ttyACM1"
WRIST_PATH="/dev/video4"
OVERHEAD_PATH="/dev/video2"
HF_USER="nirs123"
```

### Gestion compte Hugging Face

Pour uploader un dataset sur Hugging Face, il faut se connecter.

Pour cela il faut aller sur l'interface Hugging Face, se connecter et créer un token via [cette page](https://huggingface.co/settings/tokens).

On execute ensuite avec Token la commande : 

```
hf auth login --token ${HUGGINGFACE_TOKEN} --add-to-git-credential
```

On peut alors récupérer son nom de repertoire : 

```
HF_USER=$(hf auth whoami | awk -F': *' 'NR==1 {print $2}')
echo $HF_USER
```

## Téléopération

Pour lancer la téléopération :

```
uv run lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=Follower_01_DarkGreen \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=Leader_01_Purple \
    --display_data=true \
    --robot.cameras="{ \
        wrist: {type: opencv, index_or_path: ${WRIST_PATH}, width: 640, height: 480, fps: 30}, \
        overhead: {type: opencv, index_or_path: ${OVERHEAD_PATH}, width: 640, height: 480, fps: 30}}"
```

Il est possible que les fichiers de calibrage des bras ne soient pas présents. La comande `lerobot-calibrate` permet de générer ces fichiers. 

Ils sont également disponibles dans le dossier `config` (mais la commande `lerobot-calibrate` permet de générer l'arborescence de dossiers et fichiers nécessaires).

L'option `--display_data=true` permet de visualiser les données sur le logiciel Rerun, on peut charger la configuration de l'interface Rerun dans le dossier `config` avec le fichier `rerun_config.rbl`.

## Enregistrement dataset

Pour enregistrer un dataset, on peut utiliser la commande `lerobot-record`.

```
uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=Follower_01_DarkGreen \
    --teleop.type=so101_leader \
    --teleop.port=${LEADER_PORT} \
    --teleop.id=Leader_01_Purple \
    --robot.cameras="{ \
        wrist: {type: opencv, index_or_path: ${WRIST_PATH}, width: 640, height: 480, fps: 30}, \
        overhead: {type: opencv, index_or_path: ${OVERHEAD_PATH}, width: 640, height: 480, fps: 30}}" \
    --dataset.num_episodes=80 \
    --dataset.repo_id=${HF_USER}/grab_blue_highlighter_2pos4or10ep \
    --dataset.single_task="Grab the blue highlighter" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=1 
```

Explication des paramètres : 
- `--dataset.num_episodes` : nombre d'épisodes à enregistrer
- `--dataset.repo_id` : ID du dataset sur Hugging Face
- `--dataset.single_task` : Nom de la tâche à enregistrer
- `--dataset.streaming_encoding` : encodage en streaming, permet de ne pas avoir à attendre à la fin de l'enregistrement pour traiter les images.
- `--dataset.encoder_threads` : nombre de threads pour l'encodage, cela dépend de la puissance du PC.

## Entraînement modèle

Commande à executer :

```
uv run lerobot-train \
    --dataset.repo_id=${HF_USER}/blue-highlightner \
    --policy.type=act \
    --output_dir=outputs/train/act_blue_highlighter_10000 \
    --job_name=act_blue_highlighter_10000_training_job \
    --policy.device=cuda \
    --wandb.enable=false \
    --policy.repo_id=${HF_USER}/act_blue_highlighter_10000_policy \
    --steps=10000 \
    --policy.push_to_hub=true
```

On peut lancer l'entrainement sur un Google Colab avec [ces notebooks Python](https://huggingface.co/docs/lerobot/notebooks)

Avec les paramètres :
- `--dataset.repo_id` : ID du dataset sur Hugging Face sur lequel on veut entraîner le modèle
- `--policy.type` : type de politique à entraîner (voir catégorie Policies de la documentation Hugging Face LeRobot)
- `--output_dir` : dossier où enregistrer les modèles
- `--job_name` : nom du job pour wandb
- `--policy.device` : device à utiliser pour l'entraînement
- `--wandb.enable` : permet d'activer wandb
- `--policy.repo_id` : ID du modèle sur Hugging Face

Voir [documentation Hugging Face LeRobot sur la partie entraînement](https://huggingface.co/docs/lerobot/il_robots#train-a-policy)

## Evaluation de modèle

Commande à executer :

```
uv run lerobot-record  \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.cameras="{ \
        wrist: {type: opencv, index_or_path: ${WRIST_PATH}, width: 640, height: 480, fps: 30}, \
        overhead: {type: opencv, index_or_path: ${OVERHEAD_PATH}, width: 640, height: 480, fps: 30}}" \
    --robot.id=Follower_01_DarkGreen \
    --display_data=false \
    --dataset.repo_id=${HF_USER}/eval_act_blue_highlighter \
    --dataset.single_task="Grab the blue highlighter" \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=1 \
    --policy.path=${HF_USER}/act_blue_highlighter_50000_policy
    # <- Teleop optional if you want to teleoperate in between episodes \
    # --teleop.type=so101_leader \
    # --teleop.port=${LEADER_PORT} \
    # --teleop.id=Leader_01_Purple \
```

## Inférence de modèle

Voir [documentation Hugging Face LeRobot sur la partie inférence](https://huggingface.co/docs/lerobot/async)

```
uv run -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080
```

```
uv run -m lerobot.async_inference.robot_client \
    --server_address=127.0.0.1:8080 \
    --robot.type=so101_follower \
    --robot.port=${FOLLOWER_PORT} \
    --robot.id=Follower_01_DarkGreen \
    --robot.cameras="{ \
        wrist: {type: opencv, index_or_path: ${WRIST_PATH}, width: 640, height: 480, fps: 30}, \
        overhead: {type: opencv, index_or_path: ${OVERHEAD_PATH}, width: 640, height: 480, fps: 30}}" \
    --task="Grab the blue highlighter" \
    --policy_type=act \
    --pretrained_name_or_path=${HF_USER}/act_blue_highlighter_100000_policy \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5
    # --policy_device=cuda \
``` 
