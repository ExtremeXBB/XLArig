# syntax=docker/dockerfile:1

FROM --platform=$TARGETPLATFORM debian:bookworm-slim AS build

ARG DEBIAN_FRONTEND=noninteractive
ARG TARGETARCH
ARG TARGETVARIANT

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        libhwloc-dev \
        libssl-dev \
        libuv1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/xlarig
COPY . .

RUN set -eux; \
    arm_target=0; \
    case "${TARGETARCH}/${TARGETVARIANT:-}" in \
        arm64/*) arm_target=8 ;; \
        arm/v7) arm_target=7 ;; \
        amd64/*) arm_target=0 ;; \
        *) echo "Unsupported Docker target platform: ${TARGETARCH}/${TARGETVARIANT:-}" >&2; exit 1 ;; \
    esac; \
    cmake -S . -B /tmp/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DARM_TARGET="${arm_target}" \
        -DWITH_HWLOC=ON \
        -DWITH_TLS=ON; \
    cmake --build /tmp/build --parallel "$(nproc)"; \
    install -D -m 0755 /tmp/build/xlarig /out/usr/local/bin/xlarig

FROM --platform=$TARGETPLATFORM debian:bookworm-slim AS runtime

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libhwloc15 \
        libssl3 \
        libuv1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /out/usr/local/bin/xlarig /usr/local/bin/xlarig
COPY src/config.json /etc/xlarig/config.json

WORKDIR /etc/xlarig
ENTRYPOINT ["xlarig"]
CMD ["--config=/etc/xlarig/config.json"]
