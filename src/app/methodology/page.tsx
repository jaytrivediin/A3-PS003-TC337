import type { ReactNode } from "react";
import { ArrowDown } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/primitives";

function Section({ n, title, children }: { n: number; title: string; children: ReactNode }) {
  return (
    <Card id={`s${n}`}>
      <CardHeader>
        <CardTitle className="text-base">
          <span className="mr-2 inline-flex h-6 w-6 items-center justify-center rounded-md bg-blue-50 text-xs font-bold text-blue-700">{n}</span>
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 text-sm leading-relaxed text-slate-700">{children}</CardContent>
    </Card>
  );
}

function Node({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "blue" | "cyan" | "emerald" }) {
  const styles = {
    slate: "border-slate-300 bg-white text-slate-800",
    blue: "border-blue-300 bg-blue-50 text-blue-900",
    cyan: "border-cyan-300 bg-cyan-50 text-cyan-900",
    emerald: "border-emerald-300 bg-emerald-50 text-emerald-900",
  }[tone];
  return <div className={`rounded-lg border px-3 py-2 text-center text-xs font-semibold tracking-wide ${styles}`}>{children}</div>;
}

const Arrow = () => <ArrowDown className="mx-auto h-4 w-4 text-slate-400" />;

export default function MethodologyPage() {
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Methodology</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">How FedTraffic turns the PEMS04 archive into comparable centralized and federated forecasting experiments, in plain language.</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Experiment pipeline</CardTitle>
          <CardDescription>Both branches share the same data, the same chronological split, the same GRU architecture and the same evaluation windows.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mx-auto max-w-3xl">
            <div className="mx-auto flex max-w-xs flex-col gap-1">
              <Node tone="blue">PEMS04 DATA (16992 × 307 × 3)</Node>
              <Arrow />
              <Node>PREPROCESSING (float32, z-score fitted on training period)</Node>
              <Arrow />
              <Node>CHRONOLOGICAL SPLIT (70 % / 10 % / 20 %)</Node>
              <Arrow />
              <Node>SIMULATED CLIENT PARTITIONING (IID / non-IID by sensor)</Node>
            </div>
            <div className="mt-1 grid grid-cols-2 gap-6">
              <div className="flex flex-col gap-1">
                <Arrow />
                <Node tone="cyan">CENTRALIZED TRAINING (pooled windows)</Node>
              </div>
              <div className="flex flex-col gap-1">
                <Arrow />
                <Node tone="cyan">FEDERATED TRAINING</Node>
                <Arrow />
                <Node>LOCAL TRAINING (each client, E epochs)</Node>
                <Arrow />
                <Node>MODEL UPDATES (serialised state_dict, bytes measured)</Node>
                <Arrow />
                <Node>FEDAVG SERVER (weighted by n_k)</Node>
                <Arrow />
                <Node>UPDATED GLOBAL MODEL</Node>
              </div>
            </div>
            <div className="mt-1 grid grid-cols-2 gap-6">
              <div className="flex flex-col gap-1"><Arrow /><Node>EVALUATION (same validation / test windows)</Node></div>
              <div className="flex flex-col gap-1"><Arrow /><Node>EVALUATION (same validation / test windows)</Node></div>
            </div>
            <div className="mx-auto mt-1 flex max-w-xs flex-col gap-1">
              <Arrow />
              <Node tone="emerald">RESULTS AND ANALYTICS (MAE, RMSE, bytes, convergence)</Node>
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Section n={1} title="The traffic forecasting problem">
          <p>Each PEMS04 sensor reports traffic flow (vehicles per 5 minutes), occupancy and speed. Given the last hour of observations (12 steps × 3 features) for one sensor, the model predicts the flow for the next hour (12 steps).</p>
          <p>Short-term forecasts like these feed signal timing, ramp metering and traveller information. The task is a multi-step regression problem evaluated in original flow units.</p>
        </Section>
        <Section n={2} title="Centralized learning">
          <p>The conventional approach: every window is pooled on one server and a single GRU is trained with Adam on the whole pool. FedTraffic runs this as an independent experiment with the same architecture, epoch budget, seed and evaluation windows as the federated runs — it is never derived from federated metrics.</p>
          <p>Its communication cost is reported as an <em>estimated</em> one-time upload of the selected training windows (float32), clearly separated from the measured federated model traffic.</p>
        </Section>
        <Section n={3} title="Federated learning">
          <p>Instead of moving raw windows, the server sends the current global parameters to each client, clients train locally on their own windows and only send parameters back. Raw training examples never leave the simulated client.</p>
          <p>Model updates are still exchanged every round, so federated learning trades data upload for repeated model transfers; whether it is cheaper depends on model size, rounds and client count — FedTraffic measures this rather than assuming it.</p>
        </Section>
        <Section n={4} title="Simulated traffic clients">
          <p>Clients are logical participants inside one Python process (3–5 per run). Each owns a disjoint set of training windows and is trained sequentially on the CPU. They are not independent intersections or live installations, and the visualisation reflects the real state of the backend job.</p>
          <p>Normalisation statistics are computed on the global training period and shared by all clients. This is a benchmarking convenience for a fair comparison, not a privacy-preserving mechanism.</p>
        </Section>
        <Section n={5} title="IID versus non-IID distributions">
          <p><strong>IID:</strong> the sampled training windows from all sensors are shuffled with the run seed and divided into equal, duplicate-free shards. Every client sees a representative mixture of roads.</p>
          <p><strong>Non-IID (sensor-based):</strong> sensors are ranked by their training-period mean flow and split into contiguous groups; each client owns one group and samples its windows only from those sensors. Clients therefore see systematically different traffic levels (quiet roads vs. busy roads), which is a realistic source of client drift. The test period is never used to build partitions.</p>
        </Section>
        <Section n={6} title="GRU traffic forecasting">
          <p>A Gated Recurrent Unit reads the 12-step input sequence and keeps a hidden state (size 32 by default). The final hidden state passes through a linear layer that outputs the 12 future flow values. Training minimises the mean-squared error on z-scored flow; predictions are mapped back to vehicles per 5 minutes before any metric is computed.</p>
          <p>The architecture is identical for all methods and configurable (hidden size, layers, lookback, horizon).</p>
        </Section>
        <Section n={7} title="Federated Averaging (FedAvg)">
          <p>Each round: copy global parameters to every client → train locally for E epochs → collect the state dictionaries → average them with weights proportional to the local example counts → load the result as the new global model → evaluate on the validation windows.</p>
          <p className="rounded-md bg-slate-50 p-2 font-mono text-xs">W_global_next = Σ_k (n_k / Σ_j n_j) · W_k</p>
          <p>The aggregation operates on the actual PyTorch state dictionaries (float tensors averaged, integer buffers taken from the largest client) and is covered by a unit test on simple tensors.</p>
        </Section>
        <Section n={8} title="MAE and RMSE">
          <p className="rounded-md bg-slate-50 p-2 font-mono text-xs">MAE = mean(|actual − predicted|)<br />RMSE = sqrt(mean((actual − predicted)²))</p>
          <p>Both are computed in original flow units over every forecast step of every evaluation window. Lower is better; RMSE weighs large errors more heavily. Validation metrics monitor convergence; the fixed test period yields the final numbers and is not used for tuning.</p>
        </Section>
        <Section n={9} title="Communication-cost measurement">
          <p>Every transfer is measured by serialising the model state dictionary with <code>torch.save</code> into an in-memory buffer and taking its length. Per round: download bytes = payload × clients, upload bytes = Σ client payloads, round bytes = download + upload, cumulative bytes = Σ round bytes. The raw tensor size (parameters × 4 bytes) is reported alongside.</p>
          <p>These figures represent model payload serialisation only — no TCP/HTTP/TLS overhead, compression or retransmission is modelled, and no network packets are captured.</p>
        </Section>
        <Section n={10} title="Experimental limitations">
          <ul className="list-disc space-y-1 pl-5">
            <li>Prototype-scale training: windows are deterministically <em>sampled</em> (shown in every result); full-window runs are possible but slow on CPU.</li>
            <li>Clients are simulated sequentially in one process; there is no real network, device heterogeneity or stragglers.</li>
            <li>No formal differential privacy, secure aggregation or cryptographic protection is implemented. Sharing model updates does not guarantee privacy.</li>
            <li>Shared normalisation statistics and a single global test sample are benchmarking choices.</li>
            <li>Timestamps are derived from the documented collection period (Jan–Feb 2018); the archive contains no timestamps or coordinates.</li>
            <li>Results depend on the chosen budget (rounds, epochs, windows); federated learning is not assumed to be more accurate or cheaper.</li>
          </ul>
        </Section>
      </div>
    </div>
  );
}
