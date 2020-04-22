import os

# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.2/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = ")^j0f^bz$qw%7@8eblje%-s(finzy*d@dvkds=i!dej-jb45gp"

# SECURITY WARNING: don"t run with debug turned on in production!
DEBUG = True


def get_file_list(channel, level):
    
    from datetime import datetime
    
    DATA_DIRECTORY = os.path.normcase(r"C:\Users\iant\Documents\DATA\hdf5_copy")
    
    if channel.lower() in ["so", "lno", "uvis"]:
        if level in ["hdf5_level_0p1a", "hdf5_level_0p2a", "hdf5_level_0p3a"]:
            data_path = os.path.join(DATA_DIRECTORY, level)
    
            data_filenames_dates=[]
            for file_path, subfolders, files in os.walk(data_path):
                for each_filename in files:
                    if '.h5' in each_filename:
                        file_stats = os.stat(os.path.join(data_path, file_path, each_filename))
                        mod_date = datetime.utcfromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                        
                        data_filenames_dates.append([each_filename, mod_date])
            data_filenames_sorted=sorted(data_filenames_dates)
            return data_filenames_sorted
        else:
            return [["No files in level %s" %level], "-"]
    else:
        return [["No files for channel %s" %channel], "-"]



from django.conf.urls import url
from django.http import HttpResponse
#from django.urls import path

ROOT_URLCONF = __name__
TEMPLATES = [
{
    "BACKEND": "django.template.backends.django.DjangoTemplates",
    "DIRS": [os.path.join(BASE_DIR, "hdf5", "templates")],
},
]

def downloader(request):
    channel = request.GET.get("channel", "")
    level = request.GET.get("level", "")
    table_header = ["Checkbox", "File name", "Last modified"]
    table_items = get_file_list(channel, level)
    html = render_to_string("downloader.html", {"channel":channel, "level":level, "table_headers":table_header, "table_items":table_items})
    return HttpResponse(html)

def about(request):
    title = "HDF5 list"
    author = "Ian Thomas"
    html = render_to_string("about.html", {"title": title, "author": author})
    return HttpResponse(html)


urlpatterns = [
    url(r'^$', downloader, name='downloaderpage'),
    url(r'^about/$', about, name='aboutpage'),
]