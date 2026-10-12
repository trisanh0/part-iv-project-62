/**
 * TEMPO Display Day Client Application Controller.
 *
 * Coordinates WebSocket synchronization, participant pipeline tuning,
 * dual-screen role management (Grandstand vs Participant), 2D canvas race playback,
 * and empirical Pareto frontier evaluation.
 */

import { RaceCanvasRenderer } from "./canvas_renderer.js";
import { ParetoCanvasChart } from "./pareto_chart.js";

const ALGORITHM_METADATA = {
    // Extractors
    "numba_efficient": {
        title: "TEMPO (Numba)",
        desc: "JIT-compiled accelerated feature extractor",
    },
    "polars_statistics": {
        title: "Polars",
        desc: "Multi-threaded statistical extractor",
    },
    "numpy_statistical": {
        title: "NumPy",
        desc: "Vectorised statistical summary",
    },
    "tsfresh_efficient": {
        title: "tsfresh Standard",
        desc: "Exhaustive time-series benchmark library",
    },
    "tsfresh_minimal": {
        title: "tsfresh Minimal",
        desc: "Core time-series summary subset",
    },
    "tsfel": {
        title: "TSFEL",
        desc: "Sensor signal feature library",
    },

    // Selectors
    "None": {
        title: "None (All Features)",
        desc: "Retains all extracted features without filtering",
    },
    "l1": {
        title: "L1 Regularisation",
        desc: "Lasso penalty selecting non-zero weights",
    },
    "boruta": {
        title: "Boruta",
        desc: "All-relevant selection using shadow features",
    },
    "fdr": {
        title: "False Discovery Rate",
        desc: "Statistical significance threshold filter",
    },
    "subsampled_fdr": {
        title: "Subsampled FDR",
        desc: "Fast statistical significance approximation",
    },
    "variance_threshold": {
        title: "Variance Threshold",
        desc: "Drops constant and near-zero variance features",
    },
    "tree_importance": {
        title: "Tree Importance",
        desc: "Decision tree impurity ranking",
    },
    "select_k_best_anova": {
        title: "ANOVA F-Score",
        desc: "Univariate analysis of variance ranking",
    },
    "mutual_info": {
        title: "Mutual Information",
        desc: "Non-linear dependency scoring",
    },

    // Models
    "random_forest": {
        title: "Random Forest",
        desc: "Ensemble of 100 decision trees",
    },
    "logistic_regression": {
        title: "Logistic Regression",
        desc: "Fast linear classifier",
    },
    "ridge": {
        title: "Ridge Classifier",
        desc: "L2-regularised linear classifier",
    },
};

function getAlgorithmMeta(key) {
    if (ALGORITHM_METADATA[key]) {
        return ALGORITHM_METADATA[key];
    }
    const title = (key || "")
        .split("_")
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(" ");
    return {
        title: title || "Standard",
        desc: "Standard benchmark component",
    };
}

class DemoClientApp {
    constructor() {
        this.socket = null;
        this.clientId = null;
        this.state = null;
        this.myPlayer = null;
        this._hasJoined = false;
        this.currentTrackId = null;
        this.adminPin = "7000";

        this.raceRenderer = null;
        this.paretoChart = null;
        this._countdownTimer = null;

        this._initDOM();
        this._initCanvases();
        this._connectWebSocket();
        this._fetchTrackList();
    }

    _initDOM() {
        // Core View Screens
        this.screens = {
            lobby: document.getElementById("screen-lobby"),
            building: document.getElementById("screen-building"),
            racing: document.getElementById("screen-racing"),
            results: document.getElementById("screen-results"),
        };

        // Navigation and Challenge Track Banner
        this.trackNameEl = document.getElementById("track-name");
        this.trackDescEl = document.getElementById("track-desc");
        this.phaseBadgeEl = document.getElementById("phase-badge");
        this.spectatorBadgeEl = document.getElementById("spectator-badge");

        // Grandstand Lobby Elements
        this.inputNickname = document.getElementById("input-nickname");
        this.btnJoin = document.getElementById("btn-join");
        this.lobbyRacersList = document.getElementById("lobby-racers-list");
        this.btnLobbyStartRace = document.getElementById("btn-lobby-start-race");
        this.btnLobbyAddBot = document.getElementById("btn-lobby-add-bot");
        this.btnLobbyReset = document.getElementById("btn-lobby-reset");
        this.lobbyTrackSelect = document.getElementById("lobby-track-select");
        this.btnLobbyChangeTrack = document.getElementById("btn-lobby-change-track");

        // Pipeline Builder Elements
        this.builderRacerName = document.getElementById("builder-racer-name");
        this.btnLeaveHeat = document.getElementById("btn-leave-heat");
        this.selectExtractor = document.getElementById("select-extractor");
        this.selectSelector = document.getElementById("select-selector");
        this.selectModel = document.getElementById("select-model");
        this.btnReady = document.getElementById("btn-ready");
        this.btnBuilderStartRace = document.getElementById("btn-builder-start-race");
        this.builderRacersList = document.getElementById("builder-racers-list");

        // Component Accordion and Options Containers
        this.accordionItems = {
            extractor: document.querySelector('[data-accordion="extractor"]'),
            selector: document.querySelector('[data-accordion="selector"]'),
            model: document.querySelector('[data-accordion="model"]'),
        };
        this.accordionHeaders = {
            extractor: document.getElementById("btn-accordion-extractor"),
            selector: document.getElementById("btn-accordion-selector"),
            model: document.getElementById("btn-accordion-model"),
        };
        this.accordionContents = {
            extractor: document.getElementById("content-accordion-extractor"),
            selector: document.getElementById("content-accordion-selector"),
            model: document.getElementById("content-accordion-model"),
        };
        this.accordionSummaryLabels = {
            extractor: document.getElementById("label-selected-extractor"),
            selector: document.getElementById("label-selected-selector"),
            model: document.getElementById("label-selected-model"),
        };
        this.optionsGrids = {
            extractor: document.getElementById("options-extractor-grid"),
            selector: document.getElementById("options-selector-grid"),
            model: document.getElementById("options-model-grid"),
        };
        this.presetCards = document.querySelectorAll(".preset-card");

        // Machine Dynamics Bars (Pace, Grip, Weight)
        this.barPace = document.getElementById("bar-pace") || document.getElementById("bar-speed");
        this.barGrip = document.getElementById("bar-grip") || document.getElementById("bar-accuracy");
        this.barWeight = document.getElementById("bar-weight") || document.getElementById("bar-efficiency");
        this.labelPaceLevel = document.getElementById("label-pace-level") || document.getElementById("label-speed-level");
        this.labelGripLevel = document.getElementById("label-grip-level") || document.getElementById("label-accuracy-level");
        this.labelWeightLevel = document.getElementById("label-weight-level") || document.getElementById("label-efficiency-level");

        // Hidden telemetry elements for backward compatibility
        this.statLatency = document.getElementById("stat-latency");
        this.statScore = document.getElementById("stat-score");
        this.statRam = document.getElementById("stat-ram");

        // Race Canvas & Countdown Overlay
        this.countdownOverlay = document.getElementById("countdown-overlay");
        this.countdownNumber = document.getElementById("countdown-number");

        // Results Screen Elements
        this.podiumContainer = document.getElementById("podium-cards");
        this.btnNextRound = document.getElementById("btn-next-round");

        // Admin Modal Controls
        this.btnOpenAdmin = document.getElementById("btn-open-admin");
        this.adminModal = document.getElementById("admin-modal");
        this.btnCloseAdmin = document.getElementById("btn-close-admin");
        this.adminPinInput = document.getElementById("admin-pin");
        this.adminTrackSelect = document.getElementById("admin-track-select");
        this.btnAdminChangeTrack = document.getElementById("btn-admin-change-track");
        this.btnAdminAddBot = document.getElementById("btn-admin-add-bot");
        this.btnAdminStartRace = document.getElementById("btn-admin-start-race");
        this.btnAdminReset = document.getElementById("btn-admin-reset");
        this.adminDurationInput = document.getElementById("admin-duration-input");
        this.btnAdminSetDuration = document.getElementById("btn-admin-set-duration");

        this._setupEventListeners();
    }

    _setupEventListeners() {
        // Lobby Registration
        if (this.btnJoin) {
            this.btnJoin.addEventListener("click", () => this._onJoin());
        }
        if (this.inputNickname) {
            this.inputNickname.addEventListener("keydown", (e) => {
                if (e.key === "Enter") this._onJoin();
            });
        }

        // Grandstand Direct Actions
        if (this.btnLobbyStartRace) {
            this.btnLobbyStartRace.addEventListener("click", () => {
                this._send({ action: "start_race" });
            });
        }
        if (this.btnLobbyAddBot) {
            this.btnLobbyAddBot.addEventListener("click", () => {
                this._send({
                    action: "add_bot",
                    extractor: "tsfresh_efficient",
                    selector: "None",
                    model: "random_forest",
                });
            });
        }
        if (this.btnLobbyReset) {
            this.btnLobbyReset.addEventListener("click", () => {
                this._send({ action: "reset_lobby" });
            });
        }
        if (this.btnLobbyChangeTrack && this.lobbyTrackSelect) {
            this.btnLobbyChangeTrack.addEventListener("click", () => {
                const trackId = this.lobbyTrackSelect.value;
                if (trackId) {
                    this._send({ action: "change_track", track_id: trackId });
                }
            });
        }

        // Participant Builder Actions
        if (this.btnLeaveHeat) {
            this.btnLeaveHeat.addEventListener("click", () => this._onLeave());
        }
        if (this.btnReady) {
            this.btnReady.addEventListener("click", () => this._onToggleReady());
        }
        if (this.btnBuilderStartRace) {
            this.btnBuilderStartRace.addEventListener("click", () => {
                this._send({ action: "start_race" });
            });
        }

        // Accordion Stage Header Toggles
        Object.entries(this.accordionHeaders).forEach(([stage, header]) => {
            if (header) {
                header.addEventListener("click", () => this._toggleAccordion(stage));
            }
        });

        // Preset Archetype buttons
        document.querySelectorAll("[data-preset]").forEach((btn) => {
            btn.addEventListener("click", (e) => {
                const presetBtn = e.target.closest("[data-preset]");
                if (presetBtn) {
                    const preset = presetBtn.getAttribute("data-preset");
                    this._applyPreset(preset);
                }
            });
        });

        // Results View Actions
        if (this.btnNextRound) {
            this.btnNextRound.addEventListener("click", () => {
                this._send({ action: "reset_lobby" });
            });
        }

        // Admin Console Actions
        if (this.btnOpenAdmin && this.adminModal) {
            this.btnOpenAdmin.addEventListener("click", () => {
                this.adminModal.classList.remove("hidden");
                this._fetchTrackList();
            });
        }
        if (this.btnCloseAdmin && this.adminModal) {
            this.btnCloseAdmin.addEventListener("click", () => {
                this.adminModal.classList.add("hidden");
            });
        }
        if (this.btnAdminChangeTrack && this.adminTrackSelect) {
            this.btnAdminChangeTrack.addEventListener("click", () => {
                const trackId = this.adminTrackSelect.value;
                this._sendAdminAction("change_track", { track_id: trackId });
            });
        }
        if (this.btnAdminAddBot) {
            this.btnAdminAddBot.addEventListener("click", () => {
                this._sendAdminAction("add_bot", {
                    bot_name: "Benchmark: tsfresh",
                    extractor: "tsfresh_efficient",
                    selector: "None",
                    model: "random_forest",
                });
            });
        }
        if (this.btnAdminStartRace) {
            this.btnAdminStartRace.addEventListener("click", () => {
                this._sendAdminAction("start_race");
                if (this.adminModal) this.adminModal.classList.add("hidden");
            });
        }
        if (this.btnAdminReset) {
            this.btnAdminReset.addEventListener("click", () => {
                this._sendAdminAction("reset_lobby");
                if (this.adminModal) this.adminModal.classList.add("hidden");
            });
        }
        if (this.btnAdminSetDuration && this.adminDurationInput) {
            this.btnAdminSetDuration.addEventListener("click", () => {
                const sec = parseFloat(this.adminDurationInput.value);
                if (sec > 0) this._sendAdminAction("set_duration", { duration_seconds: sec });
            });
        }
    }

    _initCanvases() {
        const raceCanvas = document.getElementById("race-canvas");
        if (raceCanvas) {
            this.raceRenderer = new RaceCanvasRenderer(raceCanvas);
        }

        const paretoCanvas = document.getElementById("pareto-canvas");
        const tooltipEl = document.getElementById("pareto-tooltip");
        if (paretoCanvas) {
            this.paretoChart = new ParetoCanvasChart(paretoCanvas, tooltipEl);
        }
    }

    _connectWebSocket() {
        const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${proto}//${window.location.host}/ws/game`;

        this.socket = new WebSocket(wsUrl);

        this.socket.onopen = () => {
            // Connection established
        };

        this.socket.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                this._handleServerEvent(msg);
            } catch (err) {
                console.error("Error parsing WebSocket event:", err);
            }
        };

        this.socket.onclose = () => {
            setTimeout(() => this._connectWebSocket(), 2000);
        };
    }

    _send(payload) {
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify(payload));
        }
    }

    _handleServerEvent(event) {
        const type = event.type;
        const data = event.data;

        if (type === "INIT") {
            this.clientId = data.client_id;
            this._updateState(data.state);
        } else if (type === "STATE_UPDATE") {
            this._updateState(data);
        } else if (type === "COUNTDOWN_START") {
            this._startCountdownSequence(data.countdown_seconds || 3.0);
        } else if (type === "RACE_START") {
            this._startRacePlayback(data);
        } else if (type === "RACE_FINISH") {
            this._showResultsView(data);
        }
    }

    _updateState(state) {
        this.state = state;
        const track = state.track;

        if (this.trackNameEl) this.trackNameEl.textContent = track.display_name;
        if (this.trackDescEl) this.trackDescEl.textContent = `${track.domain} | ${track.description || "Optimization Challenge"}`;
        if (this.phaseBadgeEl) this.phaseBadgeEl.textContent = state.phase;

        // Resolve client slot
        const myActive = state.active_racers.find((r) => r.client_id === this.clientId);
        const mySpectator = state.spectators.find((r) => r.client_id === this.clientId);
        this.myPlayer = myActive || mySpectator;

        if (this.myPlayer) {
            this._hasJoined = true;
        }

        if (this.spectatorBadgeEl) {
            this.spectatorBadgeEl.style.display = mySpectator ? "inline-block" : "none";
        }

        // Repopulate algorithm dropdowns if track changed
        const trackChanged = this.currentTrackId !== track.track_id;
        if (trackChanged) {
            this.currentTrackId = track.track_id;
            this._populateSelectOptions(track.options, true);
        } else {
            this._populateSelectOptions(track.options, false);
        }

        // Synchronize view screens with room phase
        this._syncScreenPhase(state.phase);

        // Update participant builder header & ready button
        if (this.myPlayer) {
            if (this.builderRacerName) this.builderRacerName.textContent = this.myPlayer.nickname;
            if (this.btnReady) {
                this.btnReady.textContent = this.myPlayer.is_ready ? "Not ready" : "Ready";
                this.btnReady.className = this.myPlayer.is_ready ? "btn btn-secondary" : "btn btn-primary";
            }
            if (this.btnBuilderStartRace) {
                this.btnBuilderStartRace.classList.remove("hidden");
            }
            if (this.myPlayer.metrics) {
                this._updateStatsPreview(this.myPlayer.metrics);
            }
            if (this.myPlayer.pipeline) {
                this._syncActiveOptionsWithPipeline(this.myPlayer.pipeline);
            }
        }

        // Update racer lists
        this._renderLobbyRacers(state.active_racers, state.spectators);
        this._renderBuilderRacers(state.active_racers);
    }

    _syncScreenPhase(phase) {
        // Hide all screens prior to activating target screen
        Object.values(this.screens).forEach((s) => {
            if (s) s.classList.add("hidden");
        });

        if (phase === "LOBBY" || phase === "BUILDING") {
            if (this.countdownOverlay) this.countdownOverlay.classList.add("hidden");

            // Joined participants stay on the pipeline configuration screen
            // Grandstand viewers and spectators stay on the lobby screen
            if (this.myPlayer || this._hasJoined) {
                if (this.screens.building) this.screens.building.classList.remove("hidden");
            } else {
                if (this.screens.lobby) this.screens.lobby.classList.remove("hidden");
            }
        } else if (phase === "COUNTDOWN") {
            if (this.screens.racing) this.screens.racing.classList.remove("hidden");
            if (this.countdownOverlay) this.countdownOverlay.classList.remove("hidden");
            if (this.raceRenderer) {
                this.raceRenderer.resize();
            }
        } else if (phase === "RACING") {
            if (this.screens.racing) this.screens.racing.classList.remove("hidden");
            if (this.countdownOverlay) this.countdownOverlay.classList.add("hidden");
            if (this.raceRenderer) {
                this.raceRenderer.resize();
                if (!this.raceRenderer.isRunning && this.state?.active_racers) {
                    const elapsed = this.state.phase_start_time_utc ? (Date.now() - this.state.phase_start_time_utc * 1000) : 0;
                    this.raceRenderer.setupRace(
                        this.state.active_racers,
                        this.state.race_duration_seconds || 14.0,
                        Date.now() - elapsed
                    );
                    this.raceRenderer.start();
                }
            }
        } else if (phase === "RESULTS") {
            if (this.countdownOverlay) this.countdownOverlay.classList.add("hidden");
            if (this.screens.results) this.screens.results.classList.remove("hidden");
            if (this.paretoChart) {
                this.paretoChart.resize();
            }
            if (this.state?.results) {
                this._renderResults(this.state.results);
            }
        }
    }

    _populateSelectOptions(options, forceReset = false) {
        if (!options) return;

        const populate = (selectEl, items, defaultValue) => {
            if (!selectEl) return;
            if (!forceReset && selectEl.children.length > 0) return;

            const prevVal = selectEl.value;
            selectEl.innerHTML = "";
            items.forEach((item) => {
                const opt = document.createElement("option");
                opt.value = item;
                opt.textContent = item;
                selectEl.appendChild(opt);
            });

            if (items.includes(prevVal) && !forceReset) {
                selectEl.value = prevVal;
            } else if (defaultValue && items.includes(defaultValue)) {
                selectEl.value = defaultValue;
            }
        };

        populate(this.selectExtractor, options.extractors || [], "numba_efficient");
        populate(this.selectSelector, options.selectors || [], "None");
        populate(this.selectModel, options.models || [], "random_forest");

        // Render rich option cards for each accordion section
        this._renderOptionCards("extractor", options.extractors || [], this.selectExtractor?.value || "numba_efficient");
        this._renderOptionCards("selector", options.selectors || [], this.selectSelector?.value || "None");
        this._renderOptionCards("model", options.models || [], this.selectModel?.value || "random_forest");
        this._checkMatchingPreset();
    }

    _fetchTrackList() {
        fetch("/api/tracks")
            .then((res) => res.json())
            .then((data) => {
                const tracks = data.tracks || [];
                const currentId = data.current_track_id;
                const selects = [this.lobbyTrackSelect, this.adminTrackSelect].filter(Boolean);

                selects.forEach((sel) => {
                    const prevVal = sel.value;
                    sel.innerHTML = "";
                    tracks.forEach((t) => {
                        const opt = document.createElement("option");
                        opt.value = t.track_id;
                        opt.textContent = `${t.display_name} (${t.domain})`;
                        if (t.track_id === currentId) {
                            opt.selected = true;
                        }
                        sel.appendChild(opt);
                    });
                    if (prevVal && tracks.some((t) => t.track_id === prevVal) && !currentId) {
                        sel.value = prevVal;
                    }
                });
            })
            .catch((err) => console.error("Error fetching tracks list:", err));
    }

    _renderLobbyRacers(activeRacers, spectators) {
        if (!this.lobbyRacersList) return;
        this.lobbyRacersList.innerHTML = "";

        if (activeRacers.length === 0) {
            this.lobbyRacersList.innerHTML = `
                <div style="color: var(--text-secondary); font-size: 0.85rem; font-style: italic; padding: 8px 0;">
                    No competitors joined yet. Enter nickname above or add a benchmark bot below.
                </div>
            `;
            return;
        }

        activeRacers.forEach((r) => {
            const card = document.createElement("div");
            card.className = "racer-card";
            card.style.borderLeft = `4px solid ${r.avatar_color || "#38BDF8"}`;

            const isYou = r.client_id === this.clientId;
            card.innerHTML = `
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span class="racer-avatar-dot" style="background-color: ${r.avatar_color || '#38BDF8'};"></span>
                    <div>
                        <div style="font-weight: 700; font-size: 0.9rem; color: var(--text-primary);">
                            ${r.nickname} ${isYou ? '<span style="color: var(--color-accent); font-size: 0.75rem;">(You)</span>' : ""}
                            ${r.is_bot ? '<span style="color: var(--text-secondary); font-size: 0.75rem; font-weight: normal;">[BOT]</span>' : ""}
                        </div>
                        <div style="font-size: 0.75rem; color: var(--text-secondary);">
                            ${r.pipeline?.extractor || "TEMPO"} &bull; ${r.pipeline?.model || "RF"}
                        </div>
                    </div>
                </div>
                <div>
                    <span class="status-pill ${r.is_ready ? 'status-ready' : 'status-waiting'}">
                        ${r.is_ready ? "READY" : "WAITING"}
                    </span>
                </div>
            `;
            this.lobbyRacersList.appendChild(card);
        });

        if (spectators && spectators.length > 0) {
            const specNote = document.createElement("div");
            specNote.style.cssText = "font-size: 0.75rem; color: var(--text-secondary); font-style: italic; text-align: center; padding-top: 8px;";
            specNote.textContent = `${spectators.length} spectator(s) queued for next heat`;
            this.lobbyRacersList.appendChild(specNote);
        }
    }

    _renderBuilderRacers(activeRacers) {
        if (!this.builderRacersList) return;
        this.builderRacersList.innerHTML = "";

        const others = activeRacers.filter((r) => r.client_id !== this.clientId);
        if (others.length === 0) {
            this.builderRacersList.innerHTML = `
                <div style="color: var(--text-secondary); font-size: 0.8rem; font-style: italic; padding: 4px 0;">
                    No other racers in this heat yet. (A baseline bot will be matched when you launch!)
                </div>
            `;
            return;
        }

        others.forEach((r) => {
            const row = document.createElement("div");
            row.className = "racer-card";
            row.style.borderLeft = `3px solid ${r.avatar_color || "#38BDF8"}`;
            row.style.padding = "8px 12px";

            row.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="racer-avatar-dot" style="background-color: ${r.avatar_color || '#38BDF8'}; width: 10px; height: 10px;"></span>
                    <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-primary);">${r.nickname}</span>
                    <span style="font-size: 0.75rem; color: var(--text-secondary);">(${r.pipeline?.extractor || "TEMPO"})</span>
                </div>
                <span class="status-pill ${r.is_ready ? 'status-ready' : 'status-waiting'}" style="font-size: 0.65rem; padding: 2px 6px;">
                    ${r.is_ready ? "READY" : "WAITING"}
                </span>
            `;
            this.builderRacersList.appendChild(row);
        });
    }

    _onJoin() {
        const nickname = this.inputNickname ? this.inputNickname.value.trim() || "Visitor" : "Visitor";
        this._hasJoined = true;
        this._send({ action: "join", nickname });

        // Switch immediately to pipeline builder
        if (this.screens.lobby) this.screens.lobby.classList.add("hidden");
        if (this.screens.building) this.screens.building.classList.remove("hidden");
        this._onPipelineChange();
    }

    _onLeave() {
        this._send({ action: "leave" });
        this.myPlayer = null;
        this._hasJoined = false;
        this._syncScreenPhase(this.state?.phase || "LOBBY");
    }

    _onPipelineChange() {
        if (!this.selectExtractor || !this.selectSelector || !this.selectModel) return;

        const extractor = this.selectExtractor.value;
        const selector = this.selectSelector.value;
        const model = this.selectModel.value;

        this._send({
            action: "set_pipeline",
            extractor,
            selector,
            model,
        });
    }

    _setSelectIfAvailable(selectEl, value) {
        if (!selectEl) return;
        const exists = Array.from(selectEl.options).some((o) => o.value === value);
        if (exists) {
            selectEl.value = value;
        }
    }

    _toggleAccordion(targetStage) {
        Object.entries(this.accordionItems).forEach(([stage, item]) => {
            if (!item) return;
            if (stage === targetStage) {
                item.classList.toggle("open");
            } else {
                item.classList.remove("open");
            }
        });
    }

    _renderOptionCards(stage, items, currentValue) {
        const grid = this.optionsGrids[stage];
        if (!grid) return;
        grid.innerHTML = "";

        items.forEach((itemKey) => {
            const meta = getAlgorithmMeta(itemKey);
            const card = document.createElement("div");
            card.className = `option-card ${itemKey === currentValue ? "selected" : ""}`;
            card.setAttribute("data-option-key", itemKey);
            card.innerHTML = `
                <div class="option-card-title">${meta.title}</div>
                <div class="option-card-desc">${meta.desc}</div>
            `;

            card.addEventListener("click", () => {
                this._selectOption(stage, itemKey);
            });
            grid.appendChild(card);
        });

        if (this.accordionSummaryLabels[stage]) {
            const currentMeta = getAlgorithmMeta(currentValue);
            this.accordionSummaryLabels[stage].textContent = currentMeta.title;
        }
    }

    _selectOption(stage, itemKey) {
        let selectEl = null;
        if (stage === "extractor") selectEl = this.selectExtractor;
        else if (stage === "selector") selectEl = this.selectSelector;
        else if (stage === "model") selectEl = this.selectModel;

        if (selectEl) {
            selectEl.value = itemKey;
        }

        const grid = this.optionsGrids[stage];
        if (grid) {
            grid.querySelectorAll(".option-card").forEach((c) => {
                const k = c.getAttribute("data-option-key");
                c.classList.toggle("selected", k === itemKey);
            });
        }

        if (this.accordionSummaryLabels[stage]) {
            this.accordionSummaryLabels[stage].textContent = getAlgorithmMeta(itemKey).title;
        }

        this._checkMatchingPreset();
        this._onPipelineChange();
    }

    _syncActiveOptionsWithPipeline(pipeline) {
        if (!pipeline) return;

        const stages = [
            { stage: "extractor", key: pipeline.extractor, el: this.selectExtractor },
            { stage: "selector", key: pipeline.selector, el: this.selectSelector },
            { stage: "model", key: pipeline.model, el: this.selectModel },
        ];

        stages.forEach(({ stage, key, el }) => {
            if (el && key) {
                el.value = key;
            }
            const grid = this.optionsGrids[stage];
            if (grid && key) {
                grid.querySelectorAll(".option-card").forEach((c) => {
                    const k = c.getAttribute("data-option-key");
                    c.classList.toggle("selected", k === key);
                });
            }
            if (this.accordionSummaryLabels[stage] && key) {
                this.accordionSummaryLabels[stage].textContent = getAlgorithmMeta(key).title;
            }
        });

        this._checkMatchingPreset();
    }

    _checkMatchingPreset() {
        const curExt = this.selectExtractor ? this.selectExtractor.value : "";
        const curSel = this.selectSelector ? this.selectSelector.value : "";
        const curMod = this.selectModel ? this.selectModel.value : "";

        let activePreset = null;
        if (curExt === "numba_efficient" && curSel === "None" && curMod === "random_forest") {
            activePreset = "numba";
        } else if (curExt === "tsfresh_efficient" && curSel === "None" && curMod === "random_forest") {
            activePreset = "tsfresh";
        } else if (curExt === "polars_statistics" && curSel === "None" && curMod === "logistic_regression") {
            activePreset = "polars";
        }

        this.presetCards.forEach((card) => {
            const p = card.getAttribute("data-preset");
            card.classList.toggle("active", p === activePreset);
        });
    }

    _applyPreset(preset) {
        if (preset === "numba") {
            this._setSelectIfAvailable(this.selectExtractor, "numba_efficient");
            this._setSelectIfAvailable(this.selectSelector, "None");
            this._setSelectIfAvailable(this.selectModel, "random_forest");
        } else if (preset === "tsfresh") {
            this._setSelectIfAvailable(this.selectExtractor, "tsfresh_efficient");
            this._setSelectIfAvailable(this.selectSelector, "None");
            this._setSelectIfAvailable(this.selectModel, "random_forest");
        } else if (preset === "polars") {
            this._setSelectIfAvailable(this.selectExtractor, "polars_statistics");
            this._setSelectIfAvailable(this.selectSelector, "None");
            this._setSelectIfAvailable(this.selectModel, "logistic_regression");
        }

        const pipeline = {
            extractor: this.selectExtractor?.value,
            selector: this.selectSelector?.value,
            model: this.selectModel?.value,
        };
        this._syncActiveOptionsWithPipeline(pipeline);
        this._onPipelineChange();
    }

    _onToggleReady() {
        const currentReady = this.myPlayer?.is_ready || false;
        const nextReady = !currentReady;
        this._send({ action: "set_ready", ready: nextReady });

        if (this.btnReady) {
            this.btnReady.textContent = nextReady ? "Not ready" : "Ready";
            this.btnReady.className = nextReady ? "btn btn-secondary" : "btn btn-primary";
        }
    }

    _updateStatsPreview(metrics) {
        if (!metrics) return;

        // Abstract Pace calculation (log latency scale: 0.04s is ~95%, 40s is ~15%)
        const lat = Math.max(0.02, metrics.total_latency_s || 1.0);
        const logLat = Math.log10(lat);
        const minLog = -1.4; // ~0.04s
        const maxLog = 1.6;  // ~40s
        const paceRatio = 1.0 - Math.min(1.0, Math.max(0.0, (logLat - minLog) / (maxLog - minLog)));
        const pacePct = Math.round(15 + paceRatio * 80);

        // Abstract Grip calculation (normalized typical benchmark score [0.65, 1.0])
        const score = Math.max(0.0, Math.min(1.0, metrics.score || 0.8));
        const scoreRatio = Math.min(1.0, Math.max(0.0, (score - 0.65) / 0.35));
        const gripPct = Math.round(25 + scoreRatio * 72);

        // Abstract Weight calculation (spectrum from Lightweight ~250MB to Heavyweight ~3200MB)
        const ram = Math.max(100, metrics.extraction_ram_mb || 500);
        const minRam = 250;
        const maxRam = 3200;
        const weightRatio = Math.min(1.0, Math.max(0.0, (ram - minRam) / (maxRam - minRam)));
        const weightPct = Math.round(15 + weightRatio * 78);

        // Animate telemetry bar widths smoothly
        const paceEl = this.barPace || this.barSpeed;
        const gripEl = this.barGrip || this.barAccuracy;
        const weightEl = this.barWeight || this.barEfficiency;
        if (paceEl) paceEl.style.width = `${pacePct}%`;
        if (gripEl) gripEl.style.width = `${gripPct}%`;
        if (weightEl) weightEl.style.width = `${weightPct}%`;

        // Update qualitative level indicators
        const paceLabel = this.labelPaceLevel || this.labelSpeedLevel;
        const gripLabel = this.labelGripLevel || this.labelAccuracyLevel;
        const weightLabel = this.labelWeightLevel || this.labelEfficiencyLevel;

        if (paceLabel) {
            paceLabel.textContent = pacePct >= 80 ? "Top Pace" : pacePct >= 55 ? "Fast" : pacePct >= 35 ? "Moderate" : "Standard";
        }
        if (gripLabel) {
            gripLabel.textContent = gripPct >= 85 ? "Glued" : gripPct >= 65 ? "Firm" : "Loose";
        }
        if (weightLabel) {
            weightLabel.textContent = weightPct >= 75 ? "Heavyweight" : weightPct >= 45 ? "Balanced" : "Lightweight";
        }

        // Backward compatibility for hidden elements
        if (this.statLatency) this.statLatency.textContent = `${metrics.total_latency_s.toFixed(2)}s`;
        if (this.statScore) this.statScore.textContent = `${(metrics.score * 100).toFixed(1)}%`;
        if (this.statRam) this.statRam.textContent = `${metrics.extraction_ram_mb.toFixed(0)} MB`;
    }

    _startCountdownSequence(seconds) {
        Object.values(this.screens).forEach((s) => {
            if (s) s.classList.add("hidden");
        });
        if (this.screens.racing) this.screens.racing.classList.remove("hidden");
        if (this.countdownOverlay) this.countdownOverlay.classList.remove("hidden");

        if (this.raceRenderer) {
            this.raceRenderer.resize();
            if (this.state?.active_racers) {
                this.raceRenderer.setupRace(
                    this.state.active_racers,
                    this.state.race_duration_seconds || 14.0
                );
                this.raceRenderer.renderFrame(0);
            }
        }

        let remaining = Math.ceil(seconds);
        if (this.countdownNumber) this.countdownNumber.textContent = remaining;

        if (this._countdownTimer) {
            clearInterval(this._countdownTimer);
        }

        this._countdownTimer = setInterval(() => {
            remaining -= 1;
            if (remaining > 0) {
                if (this.countdownNumber) this.countdownNumber.textContent = remaining;
            } else if (remaining === 0) {
                if (this.countdownNumber) this.countdownNumber.textContent = "GO!";
            } else {
                clearInterval(this._countdownTimer);
                this._countdownTimer = null;
                if (this.countdownOverlay) this.countdownOverlay.classList.add("hidden");
            }
        }, 1000);
    }

    _startRacePlayback(data) {
        if (this._countdownTimer) {
            clearInterval(this._countdownTimer);
            this._countdownTimer = null;
        }
        if (this.countdownOverlay) this.countdownOverlay.classList.add("hidden");

        Object.values(this.screens).forEach((s) => {
            if (s) s.classList.add("hidden");
        });
        if (this.screens.racing) this.screens.racing.classList.remove("hidden");

        if (this.raceRenderer) {
            this.raceRenderer.resize();
            this.raceRenderer.setupRace(data.racers, data.duration_seconds || 14.0);
            this.raceRenderer.start();
        }
    }

    _showResultsView(data) {
        if (this.raceRenderer) {
            this.raceRenderer.stop();
        }

        Object.values(this.screens).forEach((s) => {
            if (s) s.classList.add("hidden");
        });
        if (this.screens.results) this.screens.results.classList.remove("hidden");
        this._renderResults(data);
    }

    _renderResults(results) {
        const rankings = results.rankings || [];
        if (this.podiumContainer) {
            this.podiumContainer.innerHTML = "";

            rankings.forEach((r, idx) => {
                const card = document.createElement("div");
                card.className = "podium-card";
                card.style.borderLeft = `4px solid ${r.avatar_color || "#38BDF8"}`;

                const awardsBadges = (r.awards || [])
                    .map((a) => {
                        if (a === "speed_champion") return `<span class="award-badge award-speed">Speed Champion</span>`;
                        if (a === "accuracy_champion") return `<span class="award-badge award-accuracy">Accuracy Champion</span>`;
                        if (a === "green_compute_champion") return `<span class="award-badge award-green">Green Compute Champion</span>`;
                        if (a === "pareto_champion") return `<span class="award-badge award-pareto">Pareto Overall</span>`;
                        return "";
                    })
                    .join(" ");

                const isYou = r.client_id === this.clientId;
                card.innerHTML = `
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; font-size: 1.1rem; color: var(--text-primary);">
                            #${idx + 1} ${r.nickname} ${isYou ? '<span style="color: var(--color-accent); font-size: 0.85rem;">(You)</span>' : ""}
                        </span>
                        <span style="font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.85rem; color: var(--color-accent); font-weight: 700;">
                            Score: ${(r.pareto_score || 0).toFixed(1)}
                        </span>
                    </div>
                    <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 4px;">
                        Extractor: <span style="color: var(--text-primary);">${r.pipeline?.extractor || "TEMPO"}</span> &bull; 
                        Selector: <span style="color: var(--text-primary);">${r.pipeline?.selector || "None"}</span> &bull; 
                        Model: <span style="color: var(--text-primary);">${r.pipeline?.model || "Model"}</span>
                    </div>
                    <div style="display: flex; gap: 16px; font-size: 0.8rem; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; margin-top: 6px; color: var(--text-secondary);">
                        <div>Latency: <span style="color: var(--text-primary); font-weight: 700;">${(r.metrics?.total_latency_s || 0).toFixed(2)}s</span></div>
                        <div>Accuracy: <span style="color: var(--text-primary); font-weight: 700;">${((r.metrics?.score || 0) * 100).toFixed(1)}%</span></div>
                        <div>RAM: <span style="color: var(--text-primary); font-weight: 700;">${(r.metrics?.extraction_ram_mb || 0).toFixed(0)} MB</span></div>
                    </div>
                    ${awardsBadges ? `<div style="display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px;">${awardsBadges}</div>` : ""}
                `;
                this.podiumContainer.appendChild(card);
            });
        }

        // Resize Pareto chart
        if (this.paretoChart) {
            this.paretoChart.resize();
        }

        // Fetch Pareto frontier scatter points
        const trackId = results.track_id || this.state?.track?.track_id || "beed";
        fetch(`/api/pareto/${trackId}`)
            .then((res) => res.json())
            .then((paretoData) => {
                if (this.paretoChart) {
                    this.paretoChart.setData(
                        paretoData.all_points,
                        paretoData.pareto_frontier,
                        rankings
                    );
                    this.paretoChart.resize();
                }
            })
            .catch((err) => console.error("Failed to load Pareto chart data:", err));
    }

    _sendAdminAction(action, payload = {}) {
        const pin = this.adminPinInput ? this.adminPinInput.value || this.adminPin : this.adminPin;
        fetch("/api/admin/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ pin, action, ...payload }),
        })
            .then((res) => res.json())
            .then((res) => {
                if (res.status !== "ok") {
                    alert(`Admin Action Error: ${res.detail || "Unauthorized"}`);
                }
            })
            .catch((err) => console.error("Admin request failed:", err));
    }
}

// Bootstrap application on DOM ready
window.addEventListener("DOMContentLoaded", () => {
    window.demoApp = new DemoClientApp();
});
