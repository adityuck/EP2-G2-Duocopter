%% GA-Based PID Optimiser
clear; clc;
delete(gcp("nocreate"))
parpool('local',7)

run("Main.mlx");
sim("Controller_Simulation.slx");

% =========================================================================
% HYPERPARAMETERS
% =========================================================================

POPULATION_SIZE = 10;
GENERATIONS     = 50;
MUTATION_RATE   = 0.08;

LOWER = [100,   0,   0,  1];   % [Kp, Ki, Kd, chi]
UPPER = [500, 200, 200,  8];
RANGES = UPPER - LOWER;

EC_ref = 13.55;
MSD_ref  = 0.00405;

% =========================================================================
% INITIALISATION
% =========================================================================

population = LOWER + (UPPER - LOWER) .* rand(POPULATION_SIZE, 4);

history_best_error = zeros(GENERATIONS, 1);
history_mean_error = zeros(GENERATIONS, 1);

overall_best_error = inf;
overall_best_pid   = zeros(1, 4);

% =========================================================================
% EVOLUTION LOOP
% =========================================================================
for gen = 1:GENERATIONS
    
    % --- Fitness Evaluation ---
    errors = zeros(POPULATION_SIZE, 1);
    parfor i = 1:POPULATION_SIZE
        errors(i) = evaluate_pid( ...
            population(i,1), population(i,2), ...
            population(i,3), population(i,4));
    end

    % --- Selection: sort best to worst ---
    [sorted_errors, sort_idx] = sort(errors);
    population = population(sort_idx, :);

    best_error = sorted_errors(1);
    mean_error = mean(sorted_errors);

    % --- Track global best ---
    if best_error < overall_best_error
        overall_best_error = best_error;
        overall_best_pid   = population(1, :);
    end

    history_best_error(gen) = best_error;
    history_mean_error(gen) = mean_error;

    % --- Elitism: keep top 50% ---
    num_parents = floor(POPULATION_SIZE / 2);
    parents = population(1:num_parents, :);

    % --- Crossover & Mutation ---
    next_generation = zeros(POPULATION_SIZE, 4);
    next_generation(1:num_parents, :) = parents;

    parfor i = (num_parents + 1):POPULATION_SIZE
        idx     = randperm(num_parents, 2);
        parent1 = parents(idx(1), :);
        parent2 = parents(idx(2), :);

        % Uniform crossover
        mask  = rand(1, 4) > 0.5;
        child = parent1 .* mask + parent2 .* (~mask);

        % Per-gene mutation (scaled to each gene's range)
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

    fprintf('Gen %3d  |  Best: %8.4f  |  Mean: %8.4f\n', ...
             gen, best_error, mean_error);
end

% =========================================================================
% RESULTS
% =========================================================================

fprintf('\n--- Evolution Complete ---\n');
fprintf('Best Error : %.4f\n',  overall_best_error);
fprintf('Optimal Kp : %.4f\n',  overall_best_pid(1));
fprintf('Optimal Ki : %.4f\n',  overall_best_pid(2));
fprintf('Optimal Kd : %.4f\n',  overall_best_pid(3));
fprintf('Optimal chi: %.4f\n',  overall_best_pid(4));

% =========================================================================
% CONVERGENCE PLOT
% =========================================================================

figure;
generations_vector = 1:GENERATIONS;
plot(generations_vector, history_mean_error, '-rs', 'LineWidth', 1.5, 'MarkerSize', 4);
hold on;
plot(generations_vector, history_best_error, '-bo', 'LineWidth', 1.5, 'MarkerSize', 4);
hold off;
title('GA Convergence');
xlabel('Generation');
ylabel('Cost (J)');
legend('Mean Error', 'Best Error', 'Location', 'northeast');
grid on;

% =========================================================================
% FINAL STEP RESPONSE
% =========================================================================

plot_step_response(overall_best_pid(1), overall_best_pid(2), overall_best_pid(3));


% =========================================================================
% FUNCTION: FITNESS EVALUATION
% =========================================================================

function total_error = evaluate_pid(Kp, Ki, Kd, chi)

    EC_ref = 13.55;
    MSD_ref  = 0.00405;

    simIn = Simulink.SimulationInput('Controller_Simulation');
    simIn = simIn.setVariable('Kp',  Kp);
    simIn = simIn.setVariable('Ki',  Ki);
    simIn = simIn.setVariable('Kd',  Kd);
    simIn = simIn.setVariable('chi', chi);

    function J = run_sim(input, M_val)
        input  = input.setVariable('M', M_val);
        simout = sim(input);
        J = (4 * simout.Energy.Data(end) / 7) / EC_ref + ...
            (3 * simout.MSD.Data(end)   / 7) / MSD_ref;
        if simout.ErrorMessage ~= ""
            J = 1e6;
        end
    end

    J_sin   = run_sim(simIn, 2);
    J_chirp = run_sim(simIn, 3);
    J_step  = run_sim(simIn, 4);

    total_error = 0.10 * J_sin + 0.45 * J_step + 0.45 * J_chirp;

end


% =========================================================================
% FUNCTION: PLOT STEP RESPONSE
% =========================================================================

function plot_step_response(Kp, Ki, Kd)

    s    = tf('s');
    wn   = 10;
    zeta = 0.6;

    plant          = wn^2 / (s^2 + 2*zeta*wn*s + wn^2);
    controller     = pid(Kp, Ki, Kd);
    closed_loop    = feedback(controller * plant, 1);

    figure;
    step(closed_loop);
    title(sprintf('Optimised PID  —  Kp: %.2f  Ki: %.2f  Kd: %.2f', Kp, Ki, Kd));
    grid on;

end