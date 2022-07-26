FROM debian:bullseye AS c3d

RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
        curl=7.74.0-1.3+deb11u1 \
        ca-certificates=20210119 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && mkdir /opt/c3d \
    && mkdir /download \
    && curl -s -L https://versaweb.dl.sourceforge.net/project/c3d/c3d/Experimental/c3d-1.3.0-Linux-gcc64.tar.gz --output /download/c3d.tar.gz \
    && tar -xf /download/c3d.tar.gz -C /opt/c3d --strip-components=1

FROM debian:bullseye AS niftyreg

ENV CXXFLAGS="-Wno-narrowing"
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
        ca-certificates=20210119 \
        cmake=3.18.4-2+deb11u1 \
        curl=7.74.0-1.3+deb11u1 \
        g++=4:10.2.1-1 \
        make=4.3-4.1 \
        unzip=6.0-26 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && mkdir /download \
    && mkdir /opt/niftyreg \
    && curl -s -L https://cfhcable.dl.sourceforge.net/project/niftyreg/nifty_reg-1.3.9/nifty_reg-1.3.9.zip --output /download/niftyreg.zip \
    && unzip /download/niftyreg.zip -d /download/niftyreg \
    && cmake /download/niftyreg/nifty_reg \
        -B /download/niftyreg/nifty_reg \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DBUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX=/opt/niftyreg \
    && make -C /download/niftyreg/nifty_reg \
    && make -C /download/niftyreg/nifty_reg install

FROM debian:bullseye AS requirements

COPY --from=niftyreg /opt/niftyreg /opt/niftyreg
COPY --from=c3d /opt/c3d /opt/c3d
COPY ./poetry.lock /
COPY ./pyproject.toml /
ENV PATH=/opt/niftyreg/bin:/opt/c3d/bin:$PATH
ENV LD_LIBRARY_PATH=/opt/niftyreg/lib:$LD_LIBRARY_PATH
RUN apt-get update -qq \
    && apt-get install -y -q --no-install-recommends \
        g++=4:10.2.1-1 \
        libdatrie1=0.2.13-1 \
        libgomp1=10.2.1-6 \
        python3.9=3.9.2-1 \
        python3.9-dev=3.9.2-1 \
        python3-pip=20.3.4-4+deb11u1 \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && pip install --prefer-binary --no-cache-dir \
        poetry==1.1.13 \
    && poetry config virtualenvs.create false \
    && poetry install \
    && apt-get purge -y -q \
        g++ \
        python3.9-dev \
    && apt-get --purge -y -qq autoremove

FROM requirements AS train
COPY ./afids-auto-train /opt/afids-auto-train
ENTRYPOINT ["/opt/afids-auto-train/run.py"]

FROM requirements AS apply
COPY ./afids-auto-apply /opt/afids-auto-apply
ENTRYPOINT ["/opt/afids-auto-apply/run.py"]
