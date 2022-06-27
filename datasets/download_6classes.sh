fileid="1-0raxs4PHOVjcT3QfHP3EBR5hLOFYyv6"
filename="mixed.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip mixed.zip -d mixed

fileid="1-5eHvbaOgCYSSUvDSjBMJoOiBeVRYJpr"
filename="section.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip section.zip -d section

fileid="1r7U9fWwNDghQ0Qo6cqtpqLiVqlBmv4Eu"
filename="surface.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
unzip surface.zip -d surface