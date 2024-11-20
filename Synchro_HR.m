% Synchronizing HR from Zwift and Smartwatch
clear 
clc
close all

%% Importing data
% Choosing the setup
ID = 7;
setup = "handcycle";
test = "FTP";
    
filepath = sprintf("00%d/Zwift/00%d_%s_%s", ID, ID, setup, test);
Table_Zwift_hc = readtable(filepath, 'VariableNamingRule', 'preserve');
Zwift = table2array(Table_Zwift_hc(:, 4))';

filepath = sprintf("00%d/Zwift/00%d_%s_%s_smartwatch", ID, ID, setup, test);
Table_Smartwatch_hc = readtable(filepath, 'VariableNamingRule', 'preserve');
Smartwatch = table2array(Table_Smartwatch_hc(:, 3))';

% Make the Zwift file as long as the Smartwatch file in order to be able to
% plot them together and realign them
buffer_time = length(Smartwatch) - length(Zwift);
Zwift = horzcat(Zwift, zeros(1, buffer_time));


%% Aligning data
% Finding delay in number of samples
n = length(Smartwatch);
% d = find(Smartwatch == max(Smartwatch), 1, 'last') - find(Zwift == max(Zwift), 1, 'last');
if ID == 7
    d = 30;
end
t = linspace(0, length(Smartwatch), length(Smartwatch));

% Realigning data
Zwift_or = Zwift;
Zwift = [Zwift_or(n-d:end), Zwift_or(1:n-(d+1))];

%% Plotting Heart Rate data
figure(1)
subplot(2, 1, 1)
set(gcf, 'Position', get(0, 'Screensize'));
plot(t, Smartwatch, 'r', t, Zwift, 'b'), hold on

%% Plotting Power output
power_output = table2array(Table_Zwift_hc(:, 7))';
power_output = horzcat(zeros(1, d), power_output, zeros(1, n-(d+length(power_output))));

plot(t, power_output)

%% Plotting cadence
cadence = table2array(Table_Zwift_hc(:, 5))';
cadence = horzcat(zeros(1, d), cadence, zeros(1, n-(d+length(cadence))));

plot(t, cadence)

%% Plotting avatar speed
avatar_speed = table2array(Table_Zwift_hc(:, 8))';
avatar_speed = horzcat(zeros(1, d), avatar_speed, zeros(1, n-(d+length(avatar_speed))));

plot(t, avatar_speed)


%% Defining plot features
xlabel("Time [s]"), axis tight
legend("Smartwatch [bpm]", "Chest strap [bpm]", "Power output [W]", "Cadence [rpm]", "Avatar speed [m/s]", 'Location', 'northeast')
title(sprintf("Data for %s, subject 00%d, %s", setup, ID, test))







%% Bicycle
setup = "bicycle";

filepath = sprintf("00%d/Zwift/00%d_%s_%s", ID, ID, setup, test);
Table_Zwift_bc = readtable(filepath, 'VariableNamingRule', 'preserve');
Zwift = table2array(Table_Zwift_bc(:, 4))';

filepath = sprintf("00%d/Zwift/00%d_%s_%s_smartwatch", ID, ID, setup, test);
Table_Smartwatch_bc = readtable(filepath, 'VariableNamingRule', 'preserve');
Smartwatch = table2array(Table_Smartwatch_bc(:, 3))';

% Make the Zwift file as long as the Smartwatch file in order to be able to
% plot them together and realign them
buffer_time = length(Smartwatch) - length(Zwift);
Zwift = horzcat(Zwift, zeros(1, buffer_time));


%% Aligning data
% Finding delay in number of samples
n = length(Smartwatch);
d = find(Smartwatch == max(Smartwatch), 1, 'last') - find(Zwift == max(Zwift), 1, 'last');
t = linspace(0, length(Smartwatch), length(Smartwatch));

% Realigning data
Zwift_or = Zwift;
Zwift = [Zwift_or(n-d:end), Zwift_or(1:n-(d+1))];

%% Plotting Heart Rate data
subplot(2, 1, 2)
plot(t, Smartwatch, 'r', t, Zwift, 'b'), hold on

%% Plotting Power output
power_output = table2array(Table_Zwift_bc(:, 7))';
power_output = horzcat(zeros(1, d), power_output, zeros(1, n-(d+length(power_output))));

plot(t, power_output)

%% Plotting cadence
cadence = table2array(Table_Zwift_bc(:, 5))';
cadence = horzcat(zeros(1, d), cadence, zeros(1, n-(d+length(cadence))));

plot(t, cadence)

%% Plotting avatar speed
avatar_speed = table2array(Table_Zwift_bc(:, 8))';
avatar_speed = horzcat(zeros(1, d), avatar_speed, zeros(1, n-(d+length(avatar_speed))));

plot(t, avatar_speed)


%% Defining plot features
xlabel("Time [s]"), axis tight
legend("Smartwatch [bpm]", "Chest strap [bpm]", "Power output [W]", "Cadence [rpm]", "Avatar speed [m/s]", 'Location', 'northeast')
title(sprintf("Data for %s, subject 00%d, %s", setup, ID, test))


%% Debugging
% Smartwatch_debug = table2array(Table_Smartwatch_hc(:, 3))';
% Matrix(1, :) = Smartwatch_debug;
% Zwift_debug = table2array(Table_Zwift_bc(:, 4))';
% if length(Smartwatch_debug) < length(Zwift_debug)
%     Smartwatch_debug = horzcat(Smartwatch_debug, zeros(1, length(Zwift)-length(Smartwatch_debug)));
% elseif length(Smartwatch_debug) > length(Zwift_debug)
%     Zwift_debug = horzcat(Zwift_debug, zeros(1, length(Smartwatch_debug)-length(Zwift_debug)));
% end
% Matrix(2, :) = Zwift_debug;
% d = find(Smartwatch_debug == max(Smartwatch_debug), 1, 'last') - find(Zwift_debug == max(Zwift_debug), 1, 'last');
% n = length(Smartwatch_debug);
% if d > 0
%     Matrix(2, :) = [Matrix(2, n-d:end), Matrix(2, 1:n-(d+1))];
% end
% 
% 
% figure(2)
% t = linspace(0, size(Matrix, 2), size(Matrix, 2));
% plot(t, Matrix(1, :), t, Matrix(2, :)), legend("Smartwatch hc", "Chest strap bc", "Location", "northwest")
% 
% Smartwatch = horzcat(Smartwatch, zeros(1, length(Zwift) - length(Smartwatch)));
% 
% figure(2)
% subplot(3, 1, 1)
% t = linspace(0, length(Zwift), length(Zwift));
% plot(t, Zwift), title("Zwift")
% subplot(3, 1, 2)
% t = linspace(0, length(Smartwatch), length(Smartwatch));
% plot(t, Smartwatch), title("Smartwatch")
% subplot(3, 1, 3)
% plot(t, Smartwatch, t, Zwift), title("Overlapping figures")