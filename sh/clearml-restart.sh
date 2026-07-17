# Stop 
sudo docker compose -f /opt/clearml/docker-compose.yml down

# start
sudo docker compose -f /opt/clearml/docker-compose.yml up -d

# SELinux fix (SELinux is preventing systemd-sysctl from open access on the file /etc/sysctl.d/99-clearml.conf.)
sudo ausearch -c 'systemd-sysctl' --raw | audit2allow -M my-systemdsysctl
sudo semodule -X 300 -i my-systemdsysctl.pp

# sudo docker run -d \
#   -p 27017:27017 \
#   --name mongodb \
#   --env GLIBC_TUNABLES="glibc.cpu.hwcaps=-SHSTK" \
