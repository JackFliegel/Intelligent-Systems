% This MATLAB file generates figure 1 in the paper by
% Izhikevich E.M. (2004)
% Which Model to Use For Cortical Spiking Neurons?
% use MATLAB R13 or later. November 2003. San Diego, CA

% Modified by Ali Minai

%%%%%%%%%%%%%%% regular spiking %%%%%%%%%%%%%%%%%%%%%%

steps = 1000;                  % This simulation runs for 1000 steps

a = 0.02; b = 0.2; c = -65;  d = 8;
V = -64; u = b * V;
VV = [];  uu = [];
tau = 0.25;
tspan = 0:tau:steps;  % tau is the discretization time-step
                      % tspan is the simulation interval
Rvar = 0;
T1 = 0;            % T1 is the time at which the step input rises
spike_ts = [];
I_values = [];  % Initialize an array to store I values

for I = 0:40  % I increases from 0 to 40
    for t = tspan
        if (t > T1)
            % I remains constant in this iteration
        else
            I = 0;
        end;
        V = V + tau * (0.04 * V^2 + 5 * V + 140 - u + I);
        u = u + tau * a * (b * V - u);
        if V > 30  % if this is a spike
            VV(end + 1) = 30;  % VV is the time-series of membrane potentials
            V = c;
            u = u + d;
            spike_ts = [spike_ts; 1];  % records a spike
            if t > 200
                Rvar = Rvar + 1;
            end
        else
            VV(end + 1) = V;
            spike_ts = [spike_ts; 0];  % records no spike
        end;
        uu(end + 1) = u;
    end;
    I_values = [I_values, Rvar];  % Store Rout value corresponding to this I
    Rvar = 0;  % Reset Rvar for the next I value
end;

% Export I_values to an Excel sheet (assuming you have the array2table function)
% Create a cell array of variable names
variableNames = cellstr(strcat('Rout_', num2str((0:40)')));

I_values_table = array2table(I_values, 'VariableNames', variableNames);
writetable(I_values_table, 'output.xlsx');


