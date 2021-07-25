
#install python packages
python uninstall -r /scripts/old_versions.txt
python install -r /scripts/requirements.txt

#install mrcnn
!chmod +x ./scripts/install_mrcnn.sh
sh ./scripts/install_mrcnn.sh
