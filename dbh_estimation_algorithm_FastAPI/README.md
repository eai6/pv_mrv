Source code for tree dbh esimtaiton algorithm


docker buildx build --platform=linux/amd64 -t pv_mrv_api:x86 .
docker tag pv_mrv_api:x86 eai6/pv_mrv_api:x86  
docker push eai6/pv_mrv_api:x86 



# Manual refresh
pip install --upgrade pip
pip install tensorflow==2.13.0rc0
pip install pandas
pip install "fastapi[all]"
pip install imutils
pip install opencv-python
pip install Pillow
pip install matplotlib
pip install statsmodels