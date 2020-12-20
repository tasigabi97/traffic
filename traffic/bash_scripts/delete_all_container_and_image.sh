#Ha hiba adódna az éppen futó progrrammal, és nem állna le a konténer,
# akkor nem lehetne újraindítani, mert minden konténernek egyedi név kell.
# Ezzel a scripttel ki lehet törölni a használt konténereket és képeket.
read  -n 1 -p"Delete all containers"
sudo docker container rm --force $(sudo docker container list --all --quiet)
read  -n 1 -p"Delete all images"
sudo docker image prune --all --force
