FROM debian:bullseye AS requirements

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
        build-essential=12.9 \
        curl=7.74.0-1.3+deb11u1 \
        libffi-dev=3.3-6 \
        libssl-dev=1.1.1n-0+deb11u1 \
        python3.9=3.9.2-1 \
        python3.9-dev=3.9.2-1 \
        python3-pip=20.3.4-4+deb11u1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && pip install --no-cache-dir \
        snakebids==0.6.0 \
        snakemake==7.8.0 \
        numpy==1.22.4 \
        pandas==1.4.2 \
        nilearn==0.9.1 \
        matplotlib==3.5.2 \
        svgutils==0.3.4

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
        curl=7.74.0-1.3+deb11u1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && mkdir /download \
    && mkdir /opt/c3d \
    && curl -s -L https://versaweb.dl.sourceforge.net/project/c3d/c3d/Experimental/c3d-1.3.0-Linux-gcc64.tar.gz --output /download/c3d.tar.gz \
    && tar -xf /download/c3d.tar.gz -C /opt/c3d --strip-components=1

ENV CXXFLAGS="-Wno-narrowing"

RUN mkdir /opt/niftyreg \
    && apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
        cmake=3.18.4-2+deb11u1 \
        cmake-curses-gui=3.18.4-2+deb11u1 \
        libpng-dev=1.6.37-3 \
        unzip=6.0-26 \
        zlib1g-dev=1:1.2.11.dfsg-2+deb11u1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && curl -s -L https://cfhcable.dl.sourceforge.net/project/niftyreg/nifty_reg-1.3.9/nifty_reg-1.3.9.zip --output /download/niftyreg.zip \
    && unzip /download/niftyreg.zip -d /download/niftyreg \
    && mv /download/niftyreg/nifty_reg /opt/niftyreg/src \
    && cmake /opt/niftyreg/src \
        -B /opt/niftyreg/src \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/niftyreg \
    && make -C /opt/niftyreg/src \
    && make -C /opt/niftyreg/src install

ENV PATH=/opt/niftyreg/bin:/opt/c3d/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/niftyreg/lib:$LD_LIBRARY_PATH

FROM requirements AS train
COPY ./afids-auto-train /opt/afids-auto-train
ENTRYPOINT ["/opt/afids-auto-train/run.py"]

FROM requirements AS apply
COPY ./afids-auto-apply /opt/afids-auto-apply
ENTRYPOINT ["/opt/afids-auto-apply/run.py"]
