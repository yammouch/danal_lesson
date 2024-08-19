FROM dolfinx/lab:latest
RUN apt-get update
RUN apt-get install -y dnsutils
RUN apt-get install -y net-tools
RUN apt-get install -y iputils-ping
RUN apt-get install -y x11-apps
RUN apt-get install -y vim
RUN apt-get install -y lsof
RUN apt-get install -y xdg-utils
RUN apt-get install -y netcat
RUN mkdir /usr/share/desktop-directories
#COPY hdfview_3.3.0-1_amd64.deb /
#RUN dpkg -i /hdfview_3.3.0-1_amd64.deb
#RUN rm /hdfview_3.3.0-1_amd64.deb
RUN pip install pandas
RUN pip install h5py
RUN pip install 'jupyterlab>=3' ipywidgets 'pyvista[all]' ipyvtklink
RUN pip install pcb-tools
#RUN pip install --upgrade gmsh
RUN pip install gmsh
RUN useradd -m -u 1000 yammouch -s /bin/bash
