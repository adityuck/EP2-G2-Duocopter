%% GA-Based PID Optimiser for Duocopter
clear; clc; delete(gcp("nocreate"));
% ============================================================
%% Initialise Variables for Physical Model
% ============================================================
m_struct=35.75e-03 * 2; %unit: kg
cg_structure = 15 + 5.353 * 10^(-3); % unit: m 
Kv=1880;
g = 9.81;
c = 0.15;
% Motor parameters: 
Motor_mass_small = 76*10^(-3); % unit: kg
prop_mass_small = 6*10^(-3); % unit: kg
ESC_mass_same = 40*10^(-3); % unit: kg
% counter mass: 
m_counter= 0.15; % unit: kg
miu_k=0.155; % Selected in the range 0.11-0.17 as 75% (Safty Choice)
% Cable parameters:
mass_cable=0.19;
L_cable=1;
h_cable=0.44; % anchor point measured to be 0.44 m not 0.33 m
% carriage_bracket parameters:
L_carriage_bracket = 0.2104;
m_carriage_bracket_and_cart = 0.45;% 150g cart + 300 g mounting disc
% ============================================================
%% Load Simulink Model & Base Variables
% ============================================================
% Load Initial PID Gains (Overwritten later by GA, but needed for compilation)
Kp = 250; Ki = 150; Kd = 20; chi = 2.25;
N = 10; 
M = 1;
model = 'Controller_Simulation';
load_system(model);
% Start parallel pool
parpool('local',19);
% Pre-compile the model once on the client to build safe cache files
disp('Pre-compiling model to prevent worker cache misses...');
sim(model);
% =========================================================================
%% HYPERPARAMETERS
% =========================================================================
POPULATION_SIZE = 200;
GENERATIONS     = 100;
MUTATION_RATE   = 0.08;
LOWER = [100,  150,  40,  1];   % [Kp, Ki, Kd, chi]
UPPER = [500, 300, 100,  8];
RANGES = UPPER - LOWER;
EC_ref = 13.55;
MSD_ref  = 0.00405;

% --- NEW: DIVERSITY HYPERPARAMETERS ---
TOURNAMENT_SIZE = 3;       % Small tournament = high diversity preservation
NUM_ELITES      = 2;       % Keep absolute best 2 individuals untouched
STAG_LIMIT      = 5;       % Generations without improvement before extinction
stagnation_count= 0;
last_best_error = inf;

% =========================================================================
%% INITIALISATION
% =========================================================================
population = LOWER + (UPPER - LOWER) .* rand(POPULATION_SIZE, 4);
history_best_error = zeros(GENERATIONS, 1);
history_mean_error = zeros(GENERATIONS, 1);
overall_best_error = inf;
overall_best_pid   = zeros(1, 4);

simInArray(POPULATION_SIZE * 3) = Simulink.SimulationInput(model);
% =========================================================================
%% EVOLUTION LOOP
% =========================================================================
for gen = 1:GENERATIONS
    
    % --- 1. Build the Batch of Simulations ---
    idx = 1;
    for i = 1:POPULATION_SIZE
        p_Kp  = population(i, 1);
        p_Ki  = population(i, 2);
        p_Kd  = population(i, 3);
        p_chi = population(i, 4);
        
        % Profile 2: Sine
        simInArray(idx) = Simulink.SimulationInput(model);
        simInArray(idx) = simInArray(idx).setVariable('Kp', p_Kp).setVariable('Ki', p_Ki).setVariable('Kd', p_Kd).setVariable('chi', p_chi).setVariable('M', 2);
        idx = idx + 1;
        % Profile 3: Chirp
        simInArray(idx) = Simulink.SimulationInput(model);
        simInArray(idx) = simInArray(idx).setVariable('Kp', p_Kp).setVariable('Ki', p_Ki).setVariable('Kd', p_Kd).setVariable('chi', p_chi).setVariable('M', 3);
        idx = idx + 1;
        % Profile 4: Step
        simInArray(idx) = Simulink.SimulationInput(model);
        simInArray(idx) = simInArray(idx).setVariable('Kp', p_Kp).setVariable('Ki', p_Ki).setVariable('Kd', p_Kd).setVariable('chi', p_chi).setVariable('M', 4);
        idx = idx + 1;
    end
    
    % --- 2. Run All Simulations in Parallel ---
    simOut = parsim(simInArray, 'ShowProgress', 'off', 'TransferBaseWorkspaceVariables', 'on');
    
    % --- 3. Calculate Fitness ---
    errors = zeros(POPULATION_SIZE, 1);
    idx = 1;
    for i = 1:POPULATION_SIZE
        J_sin   = calculate_cost(simOut(idx), EC_ref, MSD_ref);
        J_chirp = calculate_cost(simOut(idx+1), EC_ref, MSD_ref);
        J_step  = calculate_cost(simOut(idx+2), EC_ref, MSD_ref);
        idx = idx + 3;
        
        errors(i) = 0.2 * J_sin + 0.4 * J_step + 0.4 * J_chirp;
    end
    
    % --- 4. Evaluate Generation & Track Stagnation ---
    [sorted_errors, sort_idx] = sort(errors);
    population = population(sort_idx, :); % Index 1 is the best
    best_error = sorted_errors(1);
    mean_error = mean(sorted_errors);
    
    % Track global best
    if best_error < overall_best_error
        overall_best_error = best_error;
        overall_best_pid   = population(1, :);
    end
    history_best_error(gen) = best_error;
    history_mean_error(gen) = mean_error;
    
    % Check for stagnation (Improvement less than 1e-4)
    if best_error < (last_best_error - 1e-4)
        stagnation_count = 0;
        last_best_error = best_error;
    else
        stagnation_count = stagnation_count + 1;
    end
    
    fprintf('Gen %3d  |  Best: %8.4f  |  Mean: %8.4f\n', gen, best_error, mean_error);
    
    % --- 5. Mass Extinction Event (if stagnated) ---
    if stagnation_count >= STAG_LIMIT
        fprintf('   ---> Stagnation detected! Triggering Mass Extinction...\n');
        % Keep the elites, completely re-randomise the rest of the population
        population(NUM_ELITES+1:end, :) = LOWER + (UPPER - LOWER) .* rand(POPULATION_SIZE - NUM_ELITES, 4);
        stagnation_count = 0;
        continue; % Skip crossover/mutation for this generation to evaluate new randoms
    end
    
    % --- 6. Crossover & Mutation (Tournament Selection) ---
    next_generation = zeros(POPULATION_SIZE, 4);
    
    % Strict Elitism: Pass the absolute best straight to the next gen
    next_generation(1:NUM_ELITES, :) = population(1:NUM_ELITES, :);
    
    % Fill the rest of the population
    for i = (NUM_ELITES + 1):POPULATION_SIZE
        
        % Tournament for Parent 1
        tourn_idx1 = randi(POPULATION_SIZE, TOURNAMENT_SIZE, 1);
        [~, best_t1] = min(errors(tourn_idx1));
        parent1 = population(tourn_idx1(best_t1), :);
        
        % Tournament for Parent 2
        tourn_idx2 = randi(POPULATION_SIZE, TOURNAMENT_SIZE, 1);
        [~, best_t2] = min(errors(tourn_idx2));
        parent2 = population(tourn_idx2(best_t2), :);
        
        % Uniform crossover
        mask  = rand(1, 4) > 0.5;
        child = parent1 .* mask + parent2 .* (~mask);
        
        % Per-gene mutation
        for g = 1:4
            if rand() < MUTATION_RATE
                child(g) = child(g) + 0.1 * RANGES(g) * randn();
            end
        end
        
        % Enforce bounds
        child = max(child, LOWER);
        child = min(child, UPPER);
        next_generation(i, :) = child;
    end
    
    population = next_generation;
end
% =========================================================================
%% RESULTS
% =========================================================================
fprintf('\n--- Evolution Complete ---\n');
fprintf('Best Error : %.4f\n',  overall_best_error);
fprintf('Optimal Kp : %.4f\n',  overall_best_pid(1));
fprintf('Optimal Ki : %.4f\n',  overall_best_pid(2));
fprintf('Optimal Kd : %.4f\n',  overall_best_pid(3));
fprintf('Optimal chi: %.4f\n',  overall_best_pid(4));
% =========================================================================
% =========================================================================
%% AUTO-SAVE RESULTS
% =========================================================================
timestamp = char(datetime('now', 'Format', 'yyyyMMdd_HHmmss'));
mat_filename = sprintf('GA_Results_%s.mat', timestamp);
save(mat_filename, 'history_best_error', 'history_mean_error', ...
                   'overall_best_pid', 'overall_best_error', ...
                   'POPULATION_SIZE', 'GENERATIONS', 'MUTATION_RATE');
fprintf('Results successfully saved to MATLAB data file: %s\n', mat_filename);
%% CONVERGENCE PLOT
% =========================================================================
figure;
plot(1:GENERATIONS, history_mean_error, '-rs', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot(1:GENERATIONS, history_best_error, '-bo', 'LineWidth', 1.5, 'MarkerSize', 4);
hold off;
title('GA Convergence'); xlabel('Generation'); ylabel('Cost (J)');
legend('Mean Error', 'Best Error', 'Location', 'northeast'); grid on;
% =========================================================================
%% FINAL STEP RESPONSE
% =========================================================================
plot_step_response(overall_best_pid(1), overall_best_pid(2), overall_best_pid(3));

% =========================================================================
%% LOCAL FUNCTIONS
% =========================================================================
function J = calculate_cost(simout, EC_ref, MSD_ref)
    if simout.ErrorMessage ~= ""
        J = 1e6;
        return;
    end
    J = (3 * simout.Energy.Data(end) / 5) / EC_ref + ...
        (2 * simout.MSD.Data(end)   / 5) / MSD_ref;
end
function plot_step_response(Kp, Ki, Kd)
    s    = tf('s');
    wn   = 10;
    zeta = 0.6;
    plant          = wn^2 / (s^2 + 2*zeta*wn*s + wn^2);
    controller     = pid(Kp, Ki, Kd);
    closed_loop    = feedback(controller * plant, 1);
    figure; step(closed_loop);
    title(sprintf('Optimised PID  —  Kp: %.2f  Ki: %.2f  Kd: %.2f', Kp, Ki, Kd));
    grid on;
end