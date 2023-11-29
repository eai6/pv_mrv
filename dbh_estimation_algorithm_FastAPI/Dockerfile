# 
FROM python:3.8.18

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt

#
RUN apt-get -y update
RUN apt-get -y upgrade

# Install libgl1-mesa-glx
RUN apt-get update && apt-get install -y libgl1-mesa-glx

#
RUN pip install --upgrade pip

# 
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]