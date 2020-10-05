read  -n 1 -p"Delete all containers"
sudo docker container rm --force $(sudo docker container list --all --quiet)
read  -n 1 -p"Delete all images"
sudo docker image prune --all --force
