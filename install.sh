
#install python packages
python -m pip uninstall -r /scripts/old_versions.txt
python -m pip install -r /scripts/requirements.txt

#install mrcnn
!chmod +x ./scripts/install_mrcnn.sh
sh ./scripts/install_mrcnn.sh
