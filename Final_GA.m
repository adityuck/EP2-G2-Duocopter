%% GA-Based PID Optimiser for Duocopter  —  v5
%
%  Changes from v4:
%    - Separate EC_ref per profile (sine / chirp / step), matching the
%      per-profile ITAE_ref already in v4. A single EC_ref was incorrectly
%      normalising energy across profiles that run for different durations
%      and excite the system at very different amplitudes.
%
%  ── BEFORE RUNNING THE GA ──────────────────────────────────────────────
%  Calibrate six reference values (three ITAE + three EC). See Section 3.
%  Takes about 3 simulations (~1 minute).
%  ───────────────────────────────────────────────────────────────────────
% =========================================================================
clear; clc; delete(gcp('nocreate'));

% =========================================================================
%% 1. PHYSICAL MODEL PARAMETERS
% =========================================================================
m_struct                    = 35.75e-03 * 2;
cg_structure                = 15 + 5.353e-3;
Kv                          = 1880;
g                           = 9.81;
c                           = 0.15;

Motor_mass_small            = 76e-3;
prop_mass_small             = 6e-3;
ESC_mass_same               = 40e-3;

m_counter                   = 0.15;
miu_k                       = 0.155;

mass_cable                  = 0.19;
L_cable                     = 1;
h_cable                     = 0.44;

L_carriage_bracket          = 0.2104;
m_carriage_bracket_and_cart = 0.45;

% =========================================================================
%% 2. LOAD SIMULINK MODEL
% =========================================================================
Kp = 250;  Ki = 150;  Kd = 20;  chi = 2.25;
N  = 10;
M  = 1;

model = 'Controller_Simulation';
load_system(model);

parpool('local', 7);

disp('Pre-compiling model ...');
sim(model);
disp('Pre-compilation complete.');

% =========================================================================
%% 3. NORMALISATION REFERENCES  —  CALIBRATE BEFORE RUNNING GA
% =========================================================================
%
%  Six references: one ITAE + one EC per profile.
%  All are read from the terminal value of their respective logged signals.
%
%  Run these three blocks in the MATLAB command window with your initial
%  gains (Kp=250, Ki=150, Kd=20, chi=2.25), then paste the results below.
%
%  --- Sine (M=2) ---
%    M = 2; s = sim(model);
%    ITAE_ref_sine = s.ITAE.Data(end)
%    EC_ref_sine   = s.Energy.Data(end)
%
%  --- Chirp (M=3) ---
%    M = 3; s = sim(model);
%    ITAE_ref_chirp = s.ITAE.Data(end)
%    EC_ref_chirp   = s.Energy.Data(end)
%
%  --- Step (M=4) ---
%    M = 4; s = sim(model);
%    ITAE_ref_step = s.ITAE.Data(end)
%    EC_ref_step   = s.Energy.Data(end)
%
% -------------------------------------------------------------------------
refs.ITAE_sine   = 8.012327716915186e+02;   % <-- paste Sine  ITAE result
refs.ITAE_chirp  = 2.573174996665656e+02;   % <-- paste Chirp ITAE result
refs.ITAE_step   = 54.006697499999959;      % <-- paste Step  ITAE result

refs.EC_sine     = 9.371;                  % <-- paste Sine  Energy result
refs.EC_chirp    = 14.34;                     % <-- paste Chirp Energy result
refs.EC_step     = 14.01;                     % <-- paste Step  Energy result

% =========================================================================
%% 4. GA HYPERPARAMETERS
% =========================================================================
POPULATION_SIZE = 40;
GENERATIONS     = 50;

% Search bounds: [Kp,  Ki,  Kd,  chi]
LOWER  = [100,  150,  40,   1];
UPPER  = [500,  300, 100,   8];
RANGES = UPPER - LOWER;

% Selection
TOURNAMENT_SIZE = 3;
NUM_ELITES      = 2;

% Adaptive mutation
MUTATION_RATE      = 0.08;
MUTATION_SCALE_MAX = 0.15;
MUTATION_SCALE_MIN = 0.02;
MUTATION_DECAY     = 4;

% BLX-alpha crossover
BLX_ALPHA = 0.3;

% Stagnation / mass extinction
STAG_LIMIT     = 10;
STAG_THRESHOLD = 1e-4;

% =========================================================================
%% 5. INITIALISATION
% =========================================================================
population         = LOWER + RANGES .* rand(POPULATION_SIZE, 4);
history_best_error = zeros(GENERATIONS, 1);
history_mean_error = zeros(GENERATIONS, 1);
overall_best_error = inf;
overall_best_pid   = zeros(1, 4);

stagnation_count   = 0;
last_best_error    = inf;

% =========================================================================
%% 6. EVOLUTION LOOP
% =========================================================================
for gen = 1:GENERATIONS

    % --- 6a. Adaptive mutation scale -------------------------------------
    decay_factor   = exp(-MUTATION_DECAY * (gen - 1) / (GENERATIONS - 1));
    mutation_scale = MUTATION_SCALE_MIN + ...
                     (MUTATION_SCALE_MAX - MUTATION_SCALE_MIN) * decay_factor;

    % --- 6b. Build simulation batch  (3 profiles × population) ----------
    simInArray = repmat(Simulink.SimulationInput(model), POPULATION_SIZE * 3, 1);

    idx = 1;
    for i = 1:POPULATION_SIZE
        base_vars = {'Kp', population(i,1), 'Ki', population(i,2), ...
                     'Kd', population(i,3), 'chi', population(i,4)};

        simInArray(idx)   = set_vars(Simulink.SimulationInput(model), base_vars, 'M', 2);
        simInArray(idx+1) = set_vars(Simulink.SimulationInput(model), base_vars, 'M', 3);
        simInArray(idx+2) = set_vars(Simulink.SimulationInput(model), base_vars, 'M', 4);
        idx = idx + 3;
    end

    % --- 6c. Run all simulations in parallel -----------------------------
    simOut = parsim(simInArray, ...
                    'ShowProgress',                   'off', ...
                    'TransferBaseWorkspaceVariables',  'on');

    % --- 6d. Evaluate fitness --------------------------------------------
    errors = zeros(POPULATION_SIZE, 1);
    idx    = 1;
    for i = 1:POPULATION_SIZE
        J_sin   = calculate_cost(simOut(idx),   refs, 2);
        J_chirp = calculate_cost(simOut(idx+1), refs, 3);
        J_step  = calculate_cost(simOut(idx+2), refs, 4);
        idx = idx + 3;

        errors(i) = 0.2 * J_sin + 0.4 * J_chirp + 0.4 * J_step;
    end

    % --- 6e. Sort — reindex BOTH arrays together -------------------------
    [errors, sort_idx] = sort(errors);
    population         = population(sort_idx, :);

    best_error = errors(1);
    mean_error = mean(errors);

    if best_error < overall_best_error
        overall_best_error = best_error;
        overall_best_pid   = population(1, :);
    end

    history_best_error(gen) = best_error;
    history_mean_error(gen) = mean_error;

    % Stagnation check
    if best_error < (last_best_error - STAG_THRESHOLD)
        stagnation_count = 0;
        last_best_error  = best_error;
    else
        stagnation_count = stagnation_count + 1;
    end

    fprintf('Gen %3d  |  Best: %8.4f  |  Mean: %8.4f  |  MutScale: %.3f\n', ...
            gen, best_error, mean_error, mutation_scale);

    % --- 6f. Mass extinction on stagnation -------------------------------
    if stagnation_count >= STAG_LIMIT
        fprintf('   ---> Stagnation (%d gens). Triggering Mass Extinction ...\n', ...
                STAG_LIMIT);
        population(NUM_ELITES+1:end, :) = ...
            LOWER + RANGES .* rand(POPULATION_SIZE - NUM_ELITES, 4);
        stagnation_count = 0;
        continue;
    end

    % --- 6g. Crossover & Mutation  (BLX-alpha + adaptive Gaussian) -------
    next_generation = zeros(POPULATION_SIZE, 4);
    next_generation(1:NUM_ELITES, :) = population(1:NUM_ELITES, :);

    for i = (NUM_ELITES + 1):POPULATION_SIZE

        t1_idx  = randi(POPULATION_SIZE, TOURNAMENT_SIZE, 1);
        [~, w1] = min(errors(t1_idx));
        parent1 = population(t1_idx(w1), :);

        t2_idx  = randi(POPULATION_SIZE, TOURNAMENT_SIZE, 1);
        [~, w2] = min(errors(t2_idx));
        parent2 = population(t2_idx(w2), :);

        gene_lo = min(parent1, parent2) - BLX_ALPHA * abs(parent1 - parent2);
        gene_hi = max(parent1, parent2) + BLX_ALPHA * abs(parent1 - parent2);
        child   = gene_lo + (gene_hi - gene_lo) .* rand(1, 4);

        for g_idx = 1:4
            if rand() < MUTATION_RATE
                child(g_idx) = child(g_idx) + ...
                               mutation_scale * RANGES(g_idx) * randn();
            end
        end

        child = max(child, LOWER);
        child = min(child, UPPER);
        next_generation(i, :) = child;
    end

    population = next_generation;
end

% =========================================================================
%% 7. RESULTS
% =========================================================================
fprintf('\n=== Evolution Complete ===\n');
fprintf('Best Cost  : %.6f\n', overall_best_error);
fprintf('Optimal Kp : %.4f\n', overall_best_pid(1));
fprintf('Optimal Ki : %.4f\n', overall_best_pid(2));
fprintf('Optimal Kd : %.4f\n', overall_best_pid(3));
fprintf('Optimal chi: %.4f\n', overall_best_pid(4));

% =========================================================================
%% 8. AUTO-SAVE
% =========================================================================
timestamp    = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
mat_filename = sprintf('GA_Results_%s.mat', timestamp);
save(mat_filename, ...
     'history_best_error', 'history_mean_error', ...
     'overall_best_pid',   'overall_best_error', ...
     'refs',               'POPULATION_SIZE',    ...
     'GENERATIONS',        'MUTATION_RATE',      ...
     'BLX_ALPHA',          'LOWER',  'UPPER');
fprintf('Results saved to: %s\n', mat_filename);

% =========================================================================
%% 9. CONVERGENCE PLOT
% =========================================================================
figure('Name', 'GA Convergence');
plot(1:GENERATIONS, history_mean_error, '-rs', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot(1:GENERATIONS, history_best_error, '-bo', 'LineWidth', 1.5, 'MarkerSize', 4);
hold off;
title('GA Convergence');
xlabel('Generation');
ylabel('Composite Cost  J');
legend('Mean Cost', 'Best Cost', 'Location', 'northeast');
grid on;

% =========================================================================
%% 10. FINAL VALIDATION
% =========================================================================
disp('Running final validation with optimal gains ...');

val_profiles  = [2, 3, 4];
profile_names = {'Sine', 'Chirp', 'Step'};
val_inputs    = repmat(Simulink.SimulationInput(model), 3, 1);

for k = 1:3
    base_vars     = {'Kp', overall_best_pid(1), 'Ki', overall_best_pid(2), ...
                     'Kd', overall_best_pid(3), 'chi', overall_best_pid(4)};
    val_inputs(k) = set_vars(Simulink.SimulationInput(model), base_vars, ...
                             'M', val_profiles(k));
end

val_out = parsim(val_inputs, ...
                 'ShowProgress',                  'off', ...
                 'TransferBaseWorkspaceVariables', 'on');

figure('Name', 'Optimal PID — Validation');
colours = {'b', 'r', 'g'};

for k = 1:3
    if val_out(k).ErrorMessage ~= ""
        warning('Validation sim %d (%s) failed: %s', ...
                k, profile_names{k}, val_out(k).ErrorMessage);
        continue;
    end

    t_err  = val_out(k).Height_Error.Time;
    err    = val_out(k).Height_Error.Data;
    t_itae = val_out(k).ITAE.Time;
    itae   = val_out(k).ITAE.Data;

    subplot(3, 2, 2*k - 1);
    plot(t_err, err, colours{k}, 'LineWidth', 1.5);
    yline(0, 'k--', 'Setpoint', 'LabelHorizontalAlignment', 'left');
    title(sprintf('%s — Height Error', profile_names{k}));
    xlabel('Time (s)');  ylabel('Error (m)');  grid on;

    subplot(3, 2, 2*k);
    plot(t_itae, itae, colours{k}, 'LineWidth', 1.5);
    title(sprintf('%s — ITAE  (final = %.4f)', profile_names{k}, itae(end)));
    xlabel('Time (s)');  ylabel('ITAE');  grid on;
end

sgtitle(sprintf('Optimal PID: Kp=%.2f  Ki=%.2f  Kd=%.2f  chi=%.2f', ...
        overall_best_pid(1), overall_best_pid(2), ...
        overall_best_pid(3), overall_best_pid(4)));

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================

function simIn = set_vars(simIn, base_vars, varargin)
    for k = 1:2:numel(base_vars)
        simIn = simIn.setVariable(base_vars{k}, base_vars{k+1});
    end
    for k = 1:2:numel(varargin)
        simIn = simIn.setVariable(varargin{k}, varargin{k+1});
    end
end

function J = calculate_cost(simout, refs, M)
    %CALCULATE_COST  Composite cost: energy + ITAE + overshoot (step only).
    %
    %  Both energy and ITAE now use profile-specific references so each
    %  profile contributes equally to the composite cost regardless of
    %  differences in signal amplitude or simulation duration.

    if simout.ErrorMessage ~= "" || isempty(simout.ITAE) || isempty(simout.Energy)
        J = 1e6;
        return;
    end

    % Select profile-specific references
    switch M
        case 2
            itae_ref = refs.ITAE_sine;
            ec_ref   = refs.EC_sine;
        case 3
            itae_ref = refs.ITAE_chirp;
            ec_ref   = refs.EC_chirp;
        case 4
            itae_ref = refs.ITAE_step;
            ec_ref   = refs.EC_step;
        otherwise
            itae_ref = refs.ITAE_step;
            ec_ref   = refs.EC_step;
    end

    energy_cost = simout.Energy.Data(end) / ec_ref;
    itae_cost   = simout.ITAE.Data(end)   / itae_ref;

    % Overshoot penalty — step profile only
    % Sign flip in Height_Error = output crossed the setpoint
    overshoot_penalty = 0;
    if M == 4 && ~isempty(simout.Height_Error) && numel(simout.Height_Error.Data) > 1
        err = simout.Height_Error.Data;
        if err(1) > 0 && any(err < 0)
            overshoot_amount  = abs(min(err));
            overshoot_penalty = (overshoot_amount / abs(err(1))) * 5;
        end
    end

    J = 0.40 * energy_cost      + ...
        0.40 * itae_cost        + ...
        0.20 * overshoot_penalty;
end