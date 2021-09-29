USER=$(whoami)
nvidia-docker run -it -u ${USER} \
	-p $1:$1 \
	-p $2:$2 \
	-v /home/${USER}/workspace:/home/${USER}/workspace \
	-v /raid:/raid \
	-v /usr/share/zoneinfo:/usr/share/zoneinfo \
	-e NVIDIA_VISIBLE_DEVICES=$3 \
	--shm-size=32G \
	--name ${USER} \
	faster-rcnn:latest \
	jupyter lab --port=$2 --notebook-dir=~/workspace
