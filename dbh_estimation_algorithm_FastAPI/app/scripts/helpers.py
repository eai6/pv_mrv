import uuid
import datetime
import os


def generate_random_file_name():
    str1 =  str(uuid.uuid4())
    str2 = datetime.date.today()
    return f"{str1}@{str2}"


def removefile(filename):
    os.remove(filename)


def saveUploadfile(file_location, uploaded_file):
    with open(file_location, "wb+") as file_object:
        file_object.write(uploaded_file.file.read())