FROM projectmonai/monai:0.9.1
ARG DEBIAN_FRONTEND=noninteractive

RUN pip3 install clearml
RUN pip3 install clearml-agent
ADD clearml.conf /root

#update api
RUN apt-get update
RUN apt-get install openslide-tools -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install -y curl
RUN apt-get install -y unzip
RUN apt-get install -y libtiff5-dev

#buildopenslide
RUN apt-get install build-essential -y
RUN apt-get install -y git
RUN apt-get install autoconf -y
RUN apt-get -y install libtool
RUN apt-get install libopenjp2-7-dev -y
RUN apt-get install -y libglib2.0-dev
RUN apt-get install -y libcairo2-dev
RUN apt-get install -y libgdk-pixbuf2.0-dev
RUN apt-get install -y libxml2-dev
RUN apt-get install -y libsqlite3-dev
RUN apt-get install -y libsdl2-dev

#remove existing openslide
RUN apt-get remove libopenslide0 --purge -y
#build openslide
WORKDIR /opt/
#RUN git clone https://github.com/innovationcore/openslide.git
ADD openslide /opt/openslide
WORKDIR /opt/openslide
RUN git checkout origin/isyntax-support
RUN autoreconf --install --force --verbose
RUN ./configure
RUN make install

WORKDIR /workspace

