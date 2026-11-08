# The Nerve drag lab as one container: every spec in specs/ served by its trained network from models/served,
# on the CPU. Built for a Hugging Face Docker Space (port 7860) and for `docker run -p 7860:7860` anywhere.
#
#   docker build -t nerve-lab .
#   docker run --rm -p 7860:7860 nerve-lab        # then open http://localhost:7860

FROM rust:1-bookworm AS build
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates
# Build only the lab crate, CPU only; candle's CUDA feature is off.
RUN cargo build --release -p nerve --bin nerve

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 nerve
WORKDIR /app
COPY --from=build /src/target/release/nerve /usr/local/bin/nerve
COPY specs ./specs
COPY models/served ./models/served
USER nerve
EXPOSE 7860
CMD ["nerve", "serve", "--specs", "specs", "--models", "models/served", "--addr", "0.0.0.0:7860"]
