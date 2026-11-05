# Run the Nerve edge Worker on workerd, Cloudflare's open-source Workers runtime — no account, no wrangler.
#   npm run local        (bundles, creates .do-storage, serves on 127.0.0.1:8788)
# Weights come from ./export/<spec>/edge.json via a read-only disk service (R2 in production);
# sessions are Durable Objects persisted under ./.do-storage.
using Workerd = import "/workerd/workerd.capnp";

const config :Workerd.Config = (
  services = [
    (name = "nerve", worker = .nerve),
    (name = "weights", disk = (path = "export", writable = false)),
    (name = "do-storage", disk = (path = ".do-storage", writable = true)),
  ],
  sockets = [
    (name = "http", address = "127.0.0.1:8788", http = (), service = "nerve"),
  ],
);

const nerve :Workerd.Worker = (
  modules = [
    (name = "index.js", esModule = embed "dist/index.js"),
  ],
  compatibilityDate = "2026-10-01",
  bindings = [
    (name = "WEIGHTS", service = "weights"),
    (name = "SESSIONS", durableObjectNamespace = "NerveSession"),
    (name = "ORIGIN", text = ""),
  ],
  durableObjectNamespaces = [
    (className = "NerveSession", uniqueKey = "nerve-sessions-v1"),
  ],
  durableObjectStorage = (localDisk = "do-storage"),
);
