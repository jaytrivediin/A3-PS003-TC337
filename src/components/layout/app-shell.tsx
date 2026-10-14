"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";
import {
  BarChart3,
  BookOpen,
  Database,
  FlaskConical,
  Info,
  LayoutDashboard,
  Menu,
  Settings2,
  Waypoints,
  X,
} from "lucide-react";
import { AppStateProvider, useAppState } from "@/components/app-state";
import { Badge, Button, StatusDot } from "@/components/ui/primitives";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/dataset", label: "Dataset Explorer", icon: Database },
  { href: "/experiments", label: "Experiments", icon: FlaskConical },
  { href: "/simulator", label: "Federated Simulator", icon: Waypoints },
  { href: "/results", label: "Results & Analytics", icon: BarChart3 },
  { href: "/methodology", label: "Methodology", icon: BookOpen },
  { href: "/about", label: "Project Information", icon: Info },
];

function Logo() {
  return (
    <Link href="/" className="flex items-center gap-3">
      <span className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-blue-600 to-cyan-500 text-white shadow-sm">
        <Waypoints className="h-5 w-5" />
      </span>
      <span className="leading-tight">
        <span className="block text-base font-bold tracking-tight text-slate-900">FedTraffic</span>
        <span className="block text-[11px] text-slate-500">Federated Traffic Forecasting &amp; Communication Analyzer</span>
      </span>
    </Link>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => {
        const active = item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        return (
          <Link
            key={item.href}
            href={item.href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
              active ? "bg-blue-50 text-blue-700" : "text-slate-600 hover:bg-slate-100 hover:text-slate-900",
            )}
          >
            <item.icon className={cn("h-4 w-4", active ? "text-blue-600" : "text-slate-400")} />
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

function StatusIndicators() {
  const { backendOnline, healthLoading, dataset, activeJob } = useAppState();
  const datasetTone = !backendOnline ? "neutral" : dataset?.load_status === "loaded" ? "success" : dataset?.load_status === "error" ? "danger" : "warning";
  const datasetLabel = !backendOnline
    ? "Dataset: unknown"
    : dataset?.load_status === "loaded"
      ? `PEMS04 loaded · ${dataset.num_sensors} sensors`
      : dataset?.download.in_progress
        ? `Downloading ${dataset.download.percent ?? 0}%`
        : dataset?.load_status === "error"
          ? "Dataset error"
          : "PEMS04 not downloaded";
  return (
    <div className="flex flex-wrap items-center gap-2">
      <Badge tone={backendOnline ? "success" : healthLoading ? "neutral" : "danger"} className="gap-1.5 px-2.5 py-1">
        <StatusDot tone={backendOnline ? "success" : healthLoading ? "neutral" : "danger"} pulse={backendOnline} />
        {healthLoading && !backendOnline ? "Checking backend…" : backendOnline ? "Backend online" : "Backend offline"}
      </Badge>
      <Link href="/dataset">
        <Badge tone={datasetTone} className="gap-1.5 px-2.5 py-1">
          <Database className="h-3 w-3" />
          {datasetLabel}
        </Badge>
      </Link>
      {activeJob ? (
        <Link href="/simulator">
          <Badge tone="info" className="gap-1.5 px-2.5 py-1">
            <StatusDot tone="info" pulse />
            Training {activeJob.runs_finished + 1}/{activeJob.runs_total}
          </Badge>
        </Link>
      ) : null}
    </div>
  );
}

function Shell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="min-h-screen bg-slate-50 text-slate-900">
      {/* Desktop sidebar */}
      <aside className="fixed inset-y-0 left-0 z-30 hidden w-64 flex-col border-r border-slate-200 bg-white px-4 py-5 lg:flex">
        <Logo />
        <div className="mt-8 flex-1">
          <NavLinks />
        </div>
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-[11px] leading-relaxed text-slate-500">
          Prototype with <span className="font-medium text-slate-700">simulated clients</span> on the real PEMS04 dataset. No live
          intersections, no formal privacy guarantees.
        </div>
      </aside>

      {/* Mobile drawer */}
      {open ? (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-slate-900/40" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 bg-white p-4 shadow-xl">
            <div className="flex items-center justify-between">
              <Logo />
              <Button variant="ghost" size="icon" onClick={() => setOpen(false)} aria-label="Close menu">
                <X className="h-5 w-5" />
              </Button>
            </div>
            <div className="mt-6">
              <NavLinks onNavigate={() => setOpen(false)} />
            </div>
          </aside>
        </div>
      ) : null}

      <div className="lg:pl-64">
        <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
          <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 sm:px-6">
            <div className="flex items-center gap-3">
              <Button variant="ghost" size="icon" className="lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu">
                <Menu className="h-5 w-5" />
              </Button>
              <div className="lg:hidden">
                <Logo />
              </div>
              <div className="hidden lg:block">
                <StatusIndicators />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Link href="/experiments">
                <Button size="sm">
                  <Settings2 className="h-4 w-4" />
                  Configure experiment
                </Button>
              </Link>
            </div>
            <div className="w-full lg:hidden">
              <StatusIndicators />
            </div>
          </div>
        </header>
        <main className="mx-auto max-w-[1400px] px-4 py-6 sm:px-6">{children}</main>
      </div>
    </div>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <AppStateProvider>
      <Shell>{children}</Shell>
    </AppStateProvider>
  );
}
