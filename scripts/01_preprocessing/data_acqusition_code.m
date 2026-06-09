 function LOXY_socialDetection(experiment,subjName)
%% Experiment to test the effect of performing a secondary task on detection of social interactions in point-light displays
% J Schultz Spring 2015 - based on socialDetection.m.
%   Changed: saved number of actors shown, corrected noise dot
%   distribution, reduced Nnoise dots to [16 32 64 128], increased
%   NtrialsPerNoiselevel to 25/level (as in socialDetection3).
%
% J Schultz 27 Oct 2016: Changed the actions to have 2 angry and 2
%   happy ones, and implemented adaptive method (PSY) using Palamedes toolbox.
%
% J Schultz 24 Nov 2016: Added 1 action, implemented scanner trigger
%   detection, changed tasks again - lots available now.
% Plan for Torge: run this code in fMRI experiment, show participants all 5
%   stimuli and ask them to recognize one of the 5. Decode the other 4
%   using MVPA. Outside the scanner, run the actionRecognition task to
%   determine recognition thresholds, and the Nactors task, to determine
%   detection thresholds.
%
% J Schultz 13 Dec 2016: added 1-back task as new possibility, but too hard
%
% Based on papers by Thornton, Rensick and Shiffrar, Perception 2002 and Neri
% et al, NN 2006
%
% Several main tasks:
%   A: "How many actors are there?" (there are options for 0/2, or 0/1/2 or 1/2 actors)
%   B: "Are the 2 actors' actions congruent?"
%   C: "Which category is the action from"?
%   D: "Is this action X" (per default the last action of the list)
% Tasks A and B have been run with a secondary task in the past, not the
%   other ones though.
%
% 1 secondary task:
%   Did one of the blocks rotate?
% NOTE: this version for Wohnheim middle testroom, has hardcoded paths to
% functs and Palamedes and slowDownInterFrameGap = 1. 

%% preparation, settings, some defaults
% to work, needs Psychtoolbox, Palamedes toolbox, own functions
studyName = 'socialDetection';  % name of the study, present in the filename
fMRI = 0;                       % if 1: white dots on black background, waits for trigger, variable ISI, response collection only during stim but no interruption. 120 trials, no adaptive procedure, task=detect action No 5.
waitScreenRefreshes = 2;        % N measured screen refreshes to wait between frames; default = 2, on some Windows computers, need 3 or even 3.5.
adaptiveMethod = 0;             % otherwise cst noise levels
debug = 0;                      % if 1, goes into keyboard instead of returning when pressing escape
giveFeedback = 0;               % feedback on main task at end of trial
estimateBeta = 0;               % for the adaptive method: if 1, evaluates both threshold and slope, otherwise just threshold
swapResponses = 0;              % swaps left and right button assignments
secondTask = 0;                 % set to 1 for flipping blocks task
secondTaskNblocksFlipping = 1;  % the N blocks that will flip
Nresponses = 2;                 % can be 2- or 3-AFC
expType = 'noisy';              % expType = 'multi';
dotColor = 0;                   % if 0 = black, 255 = white; -1 = bicolor
screenColor = 127;              % by default the screen color is grey
Ntrials = 50;                  % default value, overriden if fMRI
Nrepeats4demo = 5;              % Number of times each stimulus is shown
noStimResponseTimeWindow = inf;  % Time in s after stim is gone where subject can still respond

% add functs folder to path if needed (only used in fMRI experiment)
if ~exist('datetime_js','file')
  addpath(genpath([pwd filesep 'functs']))
  addpath(genpath([fileparts(pwd) filesep 'functs']))
  addpath(genpath([fileparts(fileparts(pwd)) filesep 'functs']))
  addpath(genpath([fileparts(fileparts(fileparts(pwd))) filesep 'functs']))
end

% is this the fMRI experiment?
if fMRI; experiment = 'fMRI'; end

% which version/task?
if ~exist('experiment','var')
  experiments = {'actionRecognitionTask','fMRI','fMRIdemo','fMRI1back','fMRI1backTraining','demoNactors','demoCongruence','N Actors, single','Congruence, single'};
  defSelection = experiments{1};
  selection = bttnChoiceDialog(experiments, 'Select experiment', defSelection,...
    'Which experiment?',[ceil(length(experiments)/2) floor(length(experiments)/2)]);
  experiment = experiments{selection};
end

% is this a demo?
if strfind(experiment,'demo') 
  demo = 1;
else
  demo = 0;
end

% Ask to change defaults if not demo
if fMRI; askSubjName = 1; % always ask
else % then we don't need to ask if we already have a name or if it's a demo
  if demo; askSubjName = 0; % don't need subjName because we save nothing
  else
    if exist('subjName','var'); askSubjName = 0;
    else; askSubjName = 1; end
  end
end
if askSubjName
  prompts = {'Probandenname:'};
  inputs = inputdlg(prompts,'Eingabe',1,{'test'});
  subjName = inputs{1};
end

% Now tell participant to have some patience (from largeMessage.m code):
message = {'Bitte warten Sie ein paar Sekunden,','das Experiment wird vorbereitet.','Gleich folgen Anweisungen.'};
scr = get(0,'screensize');
pos = round([200 scr(4)*.33 scr(3)-400 scr(4)*.33]);
startMessage = figure; set(gcf,'position',pos,'menubar','none','name','The End','numbertitle','off')
set(gca,'position',[0 0 1 1],'xtick',[],'ytick',[],'color',[.8 .8 .8]);
ypos = [0:1/(length(message)+1):1]; ypos = fliplr(ypos(2:end-1));
for m = 1:length(message)
  text(.5,ypos(m),message{m},'horizontalalignment','center','fontsize',36)
end
drawnow

% test saving
%if ~exist('data','dir'); mkdir('data'); end
%if ~demo
%  % to save:
%  filenameBase = ['data' filesep studyName '-' experiment '-' subjName '-' datetime_js];
%  % save detailed data
%  dlmwrite([filenameBase '_dataEachTrial.txt'], []);
%end

%% task settings
switch experiment
  case 'fMRI'
    whichTask = 'fMRItask'; % detect action No. 5
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    scramblingCondition = 4; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none, but only 1/6 of trials scrambled
    fMRI = 1;                       % waits for trigger (not implemented yet), variable ISI, no response collection or interruption during stim
    Ntrials = 6*18;
    ISIsToUse = .5:.5:3.5; % distribution of ISIs to use, mean is 2s
    ISIs = repmat(ISIsToUse,1,ceil(Ntrials/length(ISIsToUse)));
    ISIs = shuffle(ISIs(1:Ntrials));
    screenColor = 0; % set screen color to black
    dotColor = 255; % use white dots
    noStimResponseTimeWindow = 0;
  case 'fMRIdemo'
    whichTask = 'none';
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    if Nresponses == 3
      scramblingCondition = 3; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none
    elseif Nresponses == 2
      scramblingCondition = 4;
    end
  case 'fMRI1back'
    whichTask = '1back'; % detect repetition of any actiont
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    scramblingCondition = 4; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none, but only 1/6 of trials scrambled
    fMRI = 1;                       % waits for trigger (not implemented yet), variable ISI, no response collection or interruption during stim
    giveFeedback = 0;
    Ntrials = 5*21;
    ISIsToUse = 0:.5:2; % distribution of ISIs to use, mean is 2s
    ISIs = repmat(ISIsToUse,1,ceil(Ntrials/length(ISIsToUse)));
    ISIs = shuffle(ISIs(1:Ntrials));
    screenColor = 0; % set screen color to black
    dotColor = 255; % use white dots
    noStimResponseTimeWindow = 1;
  case 'fMRI1backTraining'
    whichTask = '1back'; % detect repetition of any actiont
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    scramblingCondition = 4; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none, but only 1/6 of trials scrambled
    fMRI = 1;                       % waits for trigger (not implemented yet), variable ISI, no response collection or interruption during stim
    giveFeedback = 1;
    Ntrials = 15;
    ISIsToUse = 0:.5:2; % distribution of ISIs to use, mean is 2s
    ISIs = repmat(ISIsToUse,1,ceil(Ntrials/length(ISIsToUse)));
    ISIs = shuffle(ISIs(1:Ntrials));
    screenColor = 0; % set screen color to black
    dotColor = 255; % use white dots
    noStimResponseTimeWindow = 1;
  case 'actionRecognitionTask'
    whichTask = 'actionRecognitionTask';
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    scramblingCondition = 1; % if 1: no scrambling, 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none
    giveFeedback = 1;               % feedback on main task at end of trial
    adaptiveMethod = 1;
  case 'demoNactors'
    whichTask = 'none';
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    if Nresponses == 3
      scramblingCondition = 3; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none
    elseif Nresponses == 2
      scramblingCondition = 4;
    end
  case 'demoCongruence'
    whichTask = 'none';
    secondTask = 0;
    showIncongruent = 1; % show incongruent actions or no?
    scramblingCondition = 2; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2.
  case 'N Actors, single'
    whichTask = 'HowManyActorsAreThere?';
    secondTask = 0;
    showIncongruent = 0; % show incongruent actions or no?
    if Nresponses == 3
      scramblingCondition = 3; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none
    elseif Nresponses == 2
      scramblingCondition = 2;
    end
    giveFeedback = 0;               % feedback on main task at end of trial
    adaptiveMethod = 1;
  case 'N Actors, dual'
    whichTask = 'HowManyActorsAreThere?';
    secondTask = 1;
    showIncongruent = 0; % show incongruent actions or no?
    if Nresponses == 3
      scramblingCondition = 3; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2. if 4: 2 actors or none
    elseif Nresponses == 2
      scramblingCondition = 4;
    end
  case 'Congruence, single'
    whichTask = 'congruentActions?';
    secondTask = 0;
    showIncongruent = 1; % show incongruent actions or no?
    scramblingCondition = 2; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2.
  case 'Congruence, dual'
    whichTask = 'congruentActions?';
    secondTask = 1;
    showIncongruent = 1; % show incongruent actions or no?
    scramblingCondition = 2; % if 2: scramble no actor or 1, if 3: scramble 0, 1 or 2.
end

%% stimulus list
stimList = {...
  'I_am_angry_M_A.txt',   'I_am_angry_M_B.txt';...
  'I_am_angry_F_A.txt',   'I_am_angry_F_B.txt';...
  'I_am_happy_M_A.txt',   'I_am_happy_M_B.txt';...
  'I_am_happy_F_A.txt',   'I_am_happy_F_B.txt';...
  %  'imitate_me_F_A.txt',   'imitate_me_F_B.txt';...
  'go_overthere_F_A.txt',   'go_overthere_F_B.txt';...
  };
stimName = {...
  'Wir streiten';...
  'Wir streiten';...
  'Wir freuen uns';...
  'Wir freuen uns';...
  %  'Imitiere mich';...
  'Geh dorthin';...
  };
% for all tasks except fMRI and its demo, use only the first 4 tasks
if ~strcmpi(experiment,'fMRI') & ~strcmpi(experiment,'fMRIdemo');
  stimList = stimList(1:4,:);
  stimName = stimName(1:4);
end
% NOTE: identical names are required to identify stimuli as belonging to the same class

% stimList = {...
%   %   'I_am_angry_M_A.txt',   'I_am_angry_M_B.txt';...
%   'I_am_happy_M_A.txt',   'I_am_happy_M_B.txt';...
%   'come_closer_F_A.txt',   'come_closer_F_B.txt';...
%   %   'get_down_F_A.txt',   'get_down_F_B.txt';...
%   %   'give_me_M_A.txt',   'give_me_M_B.txt';...
%   % %   'go_overthere_F_A.txt',   'go_overthere_F_B.txt';...
%   %   'help_yourself_M_A.txt',   'help_yourself_M_B.txt';...
%   %   'imitate_me_F_A.txt',   'imitate_me_F_B.txt';...
%   % %   'look_ceiling_M_A.txt',   'look_ceiling_M_B.txt';...
%   % %   'look_floor_F_A.txt',   'look_floor_F_B.txt';...
%   % %   'move_it_M_A.txt',   'move_it_M_B.txt';...
%   % %   'move_over_F_A.txt',   'move_over_F_B.txt';...
%   %   'no_M_A.txt',   'no_M_B.txt';...
%   % %   'pick_up_F_A.txt',   'pick_up_F_B.txt';...
%   % %   'put_down_M_A.txt',   'put_down_M_B.txt';...
%   % %   'sit_down_F_A.txt',   'sit_down_F_B.txt';...
%   %   'stand_up_M_A.txt',   'stand_up_M_B.txt';...
%   % %   'stop_F_A.txt',   'stop_F_B.txt';...
%   % %   'this_tall_M_A.txt',   'this_tall_M_B.txt';...
%   % %   'which_one_F_A.txt',   'which_one_F_B.txt';...
%   };
Nstims = size(stimList,1);

% for all demos, show each stimulus Nrepeats4demo times:
if demo; Ntrials = Nstims*Nrepeats4demo; end

%% Adaptive method or constant stimuli?

if adaptiveMethod
  addpath(genpath([fileparts(pwd) filesep 'Palamedes']))
  addpath(genpath([fileparts(fileparts(pwd)) filesep 'Palamedes']))
  addpath(genpath([fileparts(fileparts(fileparts(pwd))) filesep 'Palamedes']))
  warning('off','PALAMEDES:AMPM_setupPM:priorTranspose');
  
  % ---------- Set up Psi method in Palamedes toolbox -----------------------
  NnoiseDotsAll = NaN(Ntrials,1);
  
  %Stimulus values the method can select from
  %   stimRange = 1:164; % possible N noise dots
  % stimRange = (linspace(PF([0 1 0 0],.1,'inverse'),PF([0 1 0 0],.9999,'inverse'),21));
  
  %   %Set up psi
  %   grain = 201; %grain of posterior, high numbers make method more precise at the cost of RAM and time to compute.
  %
  %   PF = @PAL_Gumbel; %assumed psychometric function
  %
  %   % [alpha beta gamma lambda] = [threshold, slope, guess, lapse]
  %   priorAlphaRange = linspace(0,200,grain);
  %   priorBetaRange =  linspace(0,1,grain); %Use log10 transformed values of beta (slope) parameter in PF
  %   priorGammaRange = 1/3;  %fixed value (using vector here would make it a free parameter)
  %   priorLambdaRange = .02; %ditto
  %   %tip: Free parameters sensibly and responsibly (as opposed to just because you can). See www.palamedestoolbox.org/understandingfitting.html.
  
  %Set up psi
  guessRate = 1/Nresponses;
  minmaxBeta = [.0625 4];
  minmaxStimrange = [.1 .9999];
  minmaxNnoiseDots = [200 4];
  
  % calculate conversion from stimRange to NnoiseDots:
  X = [minmaxStimrange;1 1]';
  y = minmaxNnoiseDots';
  [Q, R] = qr(X,0); % from regress.m
  stimRange2NnoiseDots = R\(Q'*y); % from regress.m
  
  grain = 201; %grain of posterior, high numbers make method more precise at the cost of RAM and time to compute.
  %Always check posterior after method completes [using e.g., :
  %image(PAL_Scale0to1(PM.pdf)*64)] to check whether appropriate
  %grain and parameter ranges were used.
  
  PF = @PAL_Gumbel; %assumed psychometric function
  
  %Stimulus values the method can select from
  stimRange = (linspace(PF([0 1 0 0],.1,'inverse'),PF([0 1 0 0],.9999,'inverse'),21));
  
  %Define parameter ranges to be included in posterior
  priorAlphaRange = linspace(PF([0 1 0 0],minmaxStimrange(1),'inverse'),PF([0 1 0 0],minmaxStimrange(2),'inverse'),grain);
  if estimateBeta
    priorBetaRange = linspace(log10(minmaxBeta(1)),log10(minmaxBeta(2)),grain); % Use log10 transformed values of beta (slope) parameter in PF
  else % assumes a fixed slope at the middle of the range
    priorBetaRange = log10(mean(minmaxBeta)); % Use log10 transformed values of beta (slope) parameter in PF
  end
  priorGammaRange = guessRate;  %fixed value (using vector here would make it a free parameter)
  priorLambdaRange = .02; %ditto
  
  %Initialize PM structure
  PM = PAL_AMPM_setupPM('priorAlphaRange',priorAlphaRange,...
    'priorBetaRange',priorBetaRange,...
    'priorGammaRange',priorGammaRange,...
    'priorLambdaRange',priorLambdaRange,...
    'numtrials',Ntrials,...
    'PF' , PF,...
    'stimRange',stimRange);
  
else % not adaptive method
  if fMRI || demo % then we don't show noise dots
    NnoiseDotsAll = zeros(1,Ntrials);
  else
    NoiseLevels = [16 32 64 128]; %[12 24 48 96 192 384]; % I think only multiples of 4 work...
    NtrialsPerNoiselevel = round(Ntrials/length(NoiseLevels));
    NnoiseDotsAll = Shuffle(repmat(NoiseLevels,1,NtrialsPerNoiselevel))';
  end
end

%% main experiment settings

% Determine stimulus order in advance of experiment: flat distribution
temp = shuffle(repmat(1:Nstims,1,ceil(Ntrials/Nstims)));
stimTrialOrder = temp(1:Ntrials);

% Stim duration, size and noise variables
Nframes = 100; % hardcoded for a stimulus duration of 3.3s
% waitScreenRefreshes = 2; % default Nscreen refreshes needed for waiting between frames to get accurate stimulus timing:  % stims are recorded at 30Hz, so as screen is at 60 Hz, show new stim frame every 2 screen refreshes.
% if ispc & slowDownInterFrameGap;  waitScreenRefreshes = 3.5; end; % special step to accommodate suboptimal graphics hardware that shows the stimuli too fast
stimScaling = 5;
dotSize = 10;
dotDur = 4; % in frames, that's 4*1000/30 = 133ms, close to 120ms as in NeriEtAlNN2006
NstimDots = 16; % multiple of 4, max is 24. NeriEtAlNN2006 showed 12 dots, but this is too hard here.
noiseSpread = 40; % spread of noise from center; a multiple of coordinate range
screenSize = 4; % scales the display; a multiple of coordinate range

% for secondary task:
Nblocks = 10; % the trial is divided in N periods in which the squares appear, they move in space every time. If a target is present, one (or more) of the squares flips orientation from V to H at several successive appearances; how many is specified in next line
NblocksWithTarget = 6;
targetSecTask = shuffle([ones(1,floor(Ntrials/2)) zeros(1,ceil(Ntrials/2))])'; % determines which trials have targets
boxx = 25; % mean coordinate in x of box
boxz = 25; % mean coordinate in z of box
boxLength = 8; % the length of the long side of the box. Has to be multiple of 2
boxWidth = 4; % the thickness of the line making the box
boxVar = 2; % how much the box can jump from block to block

% Determine action congruency, N actors shown, actor location swap
if ~demo
  if showIncongruent
    congruentInteraction = shuffle([ones(1,floor(Ntrials/2)) zeros(1,ceil(Ntrials/2))]); % if 1, trial has congruent action
  else
    congruentInteraction = ones(1,Ntrials); % always show congruent
  end
  if scramblingCondition == 1 % no scrambling
    showNactors = 2*ones(1,Ntrials); % number indicates how many actors will be shown unscrambled
  elseif scramblingCondition == 2 % "classic" experiment in which either 1 or 0 actor is scrambled
    showNactors = shuffle([2*ones(1,floor(Ntrials/2)) ones(1,ceil(Ntrials/2))]); % number indicates how many actors will be shown unscrambled
  elseif scramblingCondition == 3 % scramble 0, 1 or 2 actors
    N0 = ceil(Ntrials/3);
    N1 = floor(Ntrials/3);
    N2 = Ntrials-N0-N1;
    showNactors = shuffle([2*ones(1,N2) ones(1,N1) zeros(1,N0)]); % number indicates how many actors will be shown unscrambled
  elseif scramblingCondition == 4 % scramble 0 or 2 actors
    N0 = ceil(Ntrials/6); % number of trials with 0 actors shown
    N2 = Ntrials-N0; % number of trials with 2 actors shown
    showNactors = shuffle([2*ones(1,N2) zeros(1,N0)]); % number indicates how many actors will be shown unscrambled
  end
else % demo
  showNactors = 2*ones(1,Ntrials); % always show 2 actors
  if strcmpi(experiment,'demoCongruence') % for this, also show incongruent examples
    congruentInteraction = repmat([ones(1,Nstims) zeros(1,Nstims)],1,round(Ntrials/(Nstims*2))); % show each interaction, then the 2 incongruent possibilities, and repeat.
  else
    congruentInteraction = ones(size(showNactors)); % show each interaction, no incongruent possibilities.
  end
end
congruentInteraction = congruentInteraction';
showNactors = showNactors';
swapActorLocations = shuffle([ones(1,floor(Ntrials/2)) zeros(1,ceil(Ntrials/2))])'; % if 1, actors' horizontal positions are swapped

% Text variables
if ispc; lineHeight = 1.5; else; lineHeight = 1; end
scrSiz = get(0,'ScreenSize'); fontSize = round(scrSiz(4)/30);
fontName = 'Arial';

%% Keyboard input
if isempty(which('Screen')); try, addPTBtoPath; catch; end; end
KbName('UnifyKeyNames');
try escapeKey = KbName('escape'); catch, escapeKey = KbName('esc'); end
try leftButton = KbName('LeftArrow'); catch, leftButton = KbName('left'); end; leftRespName = 'linke Pfeiltaste';
try middleButton = KbName('DownArrow'); catch, middleButton = KbName('down'); end; downRespName = 'mittlere Pfeiltaste';
try rightButton = KbName('RightArrow'); catch, rightButton = KbName('right'); end; rightRespName = 'rechte Pfeiltaste';
try space = KbName('SPACE'); catch, space = KbName('space'); end;
if fMRI; leftButton = KbName('g'); leftRespName = 'gruene Taste'; rightButton = KbName('r'); rightRespName = 'rote Taste'; end

% Determine button labels depending on task and in-experiment Instructions
switch whichTask
  case 'fMRItask'
    targetActionName = stimName{end};
    leftRespLabel  = 'Ja'; leftResp = Nstims;
    rightRespLabel = 'Nein'; rightResp = 0;
    questionMainTask = [targetActionName ' ?'];
    message = sprintf([...
      'Sie werden bewegte Punkte sehen, die manchmal\n',...
      'interagierende Personen darstellen.\n',...
      'Wenn Sie Personen erkennen, muessen Sie ueberlegen,\n',...
      'welche Handlung diese ausfuehren.\n',...
      'Wenn es die "%s"-Handlung ist,\n',...
      'dann druecken Sie bitte die %s.\n',...
      'Wenn es eine andere Handlung ist,\n',...
      'dann druecken Sie bitte die %s.\n',...
      'Sie muessen antworten, bevor die Punkte verschwinden.\n',...
      ' \n',...
      'Beginnen Sie mit einem beliebigen Tastendruck.'],...
      targetActionName,leftRespName,rightRespName);
  case '1back'
    leftRespLabel  = 'Ja'; leftResp = Nstims;
    rightRespLabel = 'Nein'; rightResp = 0;
    questionMainTask = 'Wiederholt?';
    message = sprintf([...
      'Sie werden bewegte Punkte sehen, die manchmal\n',...
      'interagierende Personen darstellen.\n',...
      'Wenn Sie Personen erkennen, muessen Sie ueberlegen,\n',...
      'welche Handlung diese ausfuehren.\n',...
      'Wenn es die gleiche Handlung wie gerade davor ist,\n',...
      'dann druecken Sie bitte die %s.\n',...
      'Wenn nicht, dann druecken Sie bitte die %s.\n',...
      ' \n',...
      'Beginnen Sie mit einem beliebigen Tastendruck.'],...
      leftRespName,rightRespName);
  case 'actionRecognitionTask'
    if ~swapResponses
      leftRespLabel  = 'Streiten'; leftResp = 1;
      rightRespLabel = 'Freuen'; rightResp = 2;
    else
      leftRespLabel  = 'Freuen'; leftResp = 2;
      rightRespLabel = 'Streiten'; rightResp = 1;
    end
    questionMainTask = [leftRespLabel '  |  ' rightRespLabel];
    message = sprintf([...
      'Sie werden bewegte Punkte sehen, die interagierende Personen darstellen.\n',...
      'Sie muessen erkennen, welche Handlung diese ausfuehren.\n',...
      'Die Aufgabe ist manchmal sehr schwierig!\n',...
      ' \n',...
      'Druecken Sie die %s fuer "%s"\n',...
      'und die %s fuer "%s".\n',...
      'Andere Handlungen werden hier nicht dargestellt.\n',...
      ' \n',...
      'Beginnen Sie mit einem beliebigen Tastendruck.'],...
      leftRespName,leftRespLabel,rightRespName,rightRespLabel);
  case 'HowManyActorsAreThere?'
    if Nresponses == 3
      if ~swapResponses
        leftRespLabel  = '2 Personen'; leftResp = 2;
        downRespLabel  = '1 Person'; downResp = 1;
        rightRespLabel = '0 Person'; rightResp = 0;
      else
        leftRespLabel  = '0 Person'; leftResp = 0;
        downRespLabel  = '1 Person'; downResp = 1;
        rightRespLabel = '2 Personen'; rightResp = 2;
      end
      questionMainTask = [leftRespLabel ' | ' downRespLabel ' | ' rightRespLabel];
      message = sprintf([...
        'Sie werden bewegte Punkte sehen, in denen sich Personen verstecken.\n',...
        'Sie muessen abschaetzen, ob es zwei, eine oder keine Person(en) sind.\n',...
        'Die Aufgabe ist manchmal sehr schwierig!\n',...
        'Bitte versuchen Sie, so genau wie moeglich zu antworten.\n',...
        'Die Geschwindigkeit spielt keine Rolle.\n',...
        ' \n',...
        'Druecken Sie die %s fuer "%s"\n',...
        'die %s fuer "%s"\n',...
        'und die %s fuer "%s".\n',...
        'Beginnen Sie mit einem beliebigen Tastendruck.'],...
        leftRespName,leftRespLabel,downRespName,downRespLabel,rightRespName,rightRespLabel);
    elseif Nresponses == 2
      if ~swapResponses
        leftRespLabel  = '2 Personen'; leftResp = 2;
        rightRespLabel = '1 Person'; rightResp = 1;
      else
        leftRespLabel  = '1 Person'; leftResp = 1;
        rightRespLabel = '2 Personen'; rightResp = 2;
      end
      questionMainTask = [leftRespLabel '  |  ' rightRespLabel];
      message = sprintf([...
        'Sie werden bewegte Punkte sehen, in denen sich Personen verstecken.\n',...
        'Sie muessen abschaetzen, ob es zwei oder eine Person(en) sind.\n',...
        'Die Aufgabe ist manchmal sehr schwierig!\n',...
        'Bitte versuchen Sie, so genau wie moeglich zu antworten.\n',...
        'Die Geschwindigkeit spielt keine Rolle.\n',...
        ' \n',...
        'Druecken Sie die %s fuer "%s"\n',...
        'und die %s fuer "%s".\n',...
        'Beginnen Sie mit einem beliebigen Tastendruck.'],...
        leftRespName,leftRespLabel,rightRespName,rightRespLabel);
    end
  case 'congruentActions?'
    leftRespLabel  = 'Passend'; leftResp = 1; % if same N char, gets centered
    rightRespLabel = 'Unpassend'; rightResp = 0;
    message = sprintf([...
      'Sie werden bewegte Punkte sehen, in denen sich zwei Personen verstecken.\n',...
      'Sie muessen abschaetzen, ob deren Handlungen zusammenpassen, oder nicht.\n',...
      'Die Aufgabe ist manchmal sehr schwierig!\n',...
      'Bitte versuchen Sie, so genau wie moeglich zu antworten.\n',...
      'Die Geschwindigkeit spielt keine Rolle.\n',...
      ' \n',...
      'Druecken Sie die %s fuer "' leftRespLabel '"\n',...
      'und die %s fuer "' rightRespLabel '".\n',...
      'Beginnen Sie mit einem beliebigen Tastendruck.'],...  
      leftRespName,rightRespName);
    questionMainTask = [leftRespLabel '   |    ' rightRespLabel];
  case 'none' % demo
    questionMainTask = 'Escape -> Abbruch.';
    leftRespLabel = '';
    rightRespLabel = '';
    if strfind(experiment,'demoCongruence') % then it's the congruency detection task, show what incongruent is
      message = sprintf([...
        'Sie werden bewegte Punkte sehen, die zwei interagierende Personen darstellen.\n',...
        'Bitte praegen Sie sich diese Interaktionen gut ein,\n',...
        'da Sie sie spaeter wiedererkennen muessen.\n',...
        'Passen deren Handlungen zusammen, wird der Name der Interaktion angezeigt.\n',...
        'Passen die Handlungen nicht zusammen, wird "unpassend" angezeigt.\n',...
        ' \n',...
        'Druecken Sie die "Escape"-Taste, um die Demonstration abzubrechen.\n',...
        'Beginnen Sie mit einem beliebigen Tastendruck.']);
    else
      message = sprintf([...
        'Sie werden bewegte Punkte sehen, die zwei interagierende Personen darstellen.\n',...
        'Bitte praegen Sie sich diese Interaktionen gut ein,\n',...
        'da Sie sie spaeter wiedererkennen muessen.\n',...
        'Sie werden jede Interaktion %d mal sehen.\n',...
        ' \n',...
        'Druecken Sie die "Escape"-Taste, um die Demonstration abzubrechen.\n',...
        'Beginnen Sie mit einem beliebigen Tastendruck.'],Nrepeats4demo);
    end
end

%% start PsychToolbox
onsets = NaN(Ntrials,1);
offsets = NaN(Ntrials,1);
try % start PsychToolbox window
  Screen('Preference', 'SkipSyncTests', 1)
  close(startMessage)
  drawnow
  
  % If there are multiple displays guess that one without the menu bar is the
  % best choice.  Dislay 0 has the menu bar.
  screens = Screen('Screens');  screenNumber = max(screens);
  [win,rect] = Screen('OpenWindow', screenNumber, screenColor);%,rect] [,pixelSize] [,numberOfBuffers] [,stereomode] [,multisample][,imagingmode][,specialFlags][,clientRect]);
  
  % Enable alpha blending with proper blend-function. We need it
  % for drawing of smoothed points:
  Screen('BlendFunction', win, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  [center(1), center(2)] = RectCenter(rect);
  ifi = Screen('GetFlipInterval', win); % duration of a frame, 1/fps
  
  Screen('TextFont', win, fontName);
  Screen('TextSize', win, fontSize);
  normBoundsRect = Screen('TextBounds', win, questionMainTask);
  questionBounds = CenterRect(normBoundsRect, rect);
  
  % Show instructions to subject, centered and in white:
  DrawFormattedText(win, message, 'center', 'center', WhiteIndex(win), [], [], [], lineHeight);  Screen('Flip', win);
  
  % Wait for button press to start:
  if ~fMRI
    KeyIsDown = 0;  while ~KeyIsDown;    [KeyIsDown, ~, KeyCode]=KbCheck;  end
    vbl = Screen('Flip', win);
  else % for fMRI, waits for button press to show Ready to scan screen
    KeyIsDown = 0;  while ~KeyIsDown;    [KeyIsDown, ~, KeyCode]=KbCheck;  end
    vbl = Screen('Flip', win);
    % *************** FMRI TRIGGER WAITING HERE ***************
    % triggers and buttons
    startAtTriggerNo = 6;
    triggerCode = KbName('t');
    DrawFormattedText(win, ['Waiting for trigger\n start at trigger no.: ' num2str(startAtTriggerNo)], 'center', 'center', WhiteIndex(win), [], [], [], lineHeight); %center(1)*.75, center(2)*.3, 255);
    vbl = Screen('Flip', win); % now ready for trigger
    nTriggersB4start = startAtTriggerNo; keyIsDown = 0; trigTime = [];
    while nTriggersB4start
      while ~keyIsDown; [keyIsDown,keyTime,keyCode] = KbCheck; end % waits for trigger pulse
      if any(keyCode(triggerCode))
        trigTime = [trigTime keyTime-vbl];
        nTriggersB4start = nTriggersB4start - 1;
        DrawFormattedText(win, num2str(nTriggersB4start), 'center', 'center', WhiteIndex(win)); %center(1)*.75, center(2)*.3, 255);
        Screen('Flip', win);
      end
      while keyIsDown; [keyIsDown,keyTime,keyCode] = KbCheck; end % waits for end of trigger pulse to continue
    end
  end
  startExpt = Screen('Flip', win);
  HideCursor
  
  switch expType % multi-pt-light displays
    
    case 'noisy' % pt-light displays with noise
      [resp,RT,resp2,RT2,perfCorrect,perfCorrectSecTask,pauseFlag] = deal(NaN(Ntrials,1));
      stimsSelected = NaN(Ntrials,2);
      
      if ~demo
        switch whichTask
          case 'HowManyActorsAreThere?'
            targetMain = showNactors;
          case 'congruentActions?'
            targetMain = congruentInteraction;
            showNactors = ones(1,Ntrials); % always show 2 actors!
          case 'fMRItask'
            targetFMRI = Nstims;
            targetMain = targetFMRI*ones(1,Ntrials);
            targetMain(showNactors==0) = NaN; % no answer needed if stimulus is scrambled
          case '1back' % for 1-back task, use stimulus index from previous trial as target
            targetMain = [NaN diff(stimTrialOrder)==0]; % press "same" if same action as just before.
            targetMain(showNactors==0) = NaN; % no answer needed if stimulus is scrambled
            targetMain(find(showNactors==0)+1) = NaN; % no answer needed if trial before was scrambled
          case 'actionRecognitionTask'
            targetMain = NaN*ones(1,Ntrials);
        end
      else
        targetMain = ones(1,Ntrials);
      end
      % trial loop
      for jj = 1:Ntrials % run Ntrials, or less if staircase and Nreversals reached
        pauseFlag(jj) = 0;
        
        % determine which stimuli to load
        if demo
          indA = rem(jj,Nstims); if indA == 0; indA = Nstims; end
        else
          indA = stimTrialOrder(jj); % select stimulus from the list
        end
        if congruentInteraction(jj) % then the 2nd actor should come from the same action
          indB = indA;
        else % incongruent: actor B is from another stimulus class
          [stimClasses,~,stimClassNo] = unique(stimName);
          Nclass = length(stimClasses);
          indBclass = setxor(1:Nclass,stimClassNo(indA));
          indB = sampleFrom(1,find(stimClassNo==indBclass));
        end
        
        stimsSelected(jj,:) = [indA indB];
        stimsAllTrials{jj,1} = stimList{indA,1};
        stimsAllTrials{jj,2} = stimList{indB,2};
        stimFileA = ['stims' filesep stimList{indA,1}];
        stimFileB = ['stims' filesep stimList{indB,2}];
        [ax,ay,az] = textread(stimFileA,'%f%f%f','delimiter','\t','headerlines',1);
        [bx,by,bz] = textread(stimFileB,'%f%f%f','delimiter','\t','headerlines',1);
        
        % determine amount of noise
        if adaptiveMethod % then determine NnoiseDots with PSY method
          if ~demo % &&  jj > 5 % then do PSY
            NnoiseDots = [PM.xCurrent 1]*stimRange2NnoiseDots;
            NnoiseDots = round(NnoiseDots/4)*4;
            % Limit to range of N noise dots allowed
            if NnoiseDots > max(minmaxNnoiseDots); NnoiseDots = max(minmaxNnoiseDots);
            elseif NnoiseDots < min(minmaxNnoiseDots); NnoiseDots = min(minmaxNnoiseDots);
            end
          else
            NnoiseDots = 0;
          end
          % accumulate N noise dots
          NnoiseDotsAll(jj) = NnoiseDots;
        else % not staircase
          reversals = 0; % keep them at 0 to avoid stopping early
          NnoiseDots = NnoiseDotsAll(jj);
        end
        % override everything if demo and if fMRI:
        if demo, NnoiseDots = 0; end
        if fMRI, NnoiseDots = 0; end
        
        % for action recognition task, determine target now;
        if strcmpi(whichTask,'actionRecognitionTask') && isnan(targetMain(jj))
          targetMain(jj) = ceil(indA/2);
        end
        
        % reshape to have 13ptLights x Nframes matrices:
        NdotsPerActor = 13;
        ax = reshape(ax,NdotsPerActor,[]);        az = reshape(az,NdotsPerActor,[]);
        bx = reshape(bx,NdotsPerActor,[]);        bz = reshape(bz,NdotsPerActor,[]);
        
        if ~exist('Nframes' ,'var')
          Nframes = min([size(ax,2) size(bx,2)]);
        end
        ax = ax(:,1:Nframes);        az = az(:,1:Nframes);
        bx = bx(:,1:Nframes);        bz = bz(:,1:Nframes);
        Ndots = NdotsPerActor*2;
%         sca; keyboard
        % calculate range of dots to use as reference for window size
        limitsx = [min([ax(:); bx(:)]) max([ax(:); bx(:)])];
        screenSizex = limitsx*screenSize;
        limitsz = [min([az(:); bz(:)]) max([az(:); bz(:)])];
        screenSizez = limitsz*screenSize;
        
        % Now, determine which stimulus dots to show when.
        % I want to not change all of them as the same time to avoid clear
        % motion transients (NeriNN2006). Here we go.
        NdotsToShow = NstimDots; setSize = NdotsToShow/dotDur; % as framerate is 30Hz, I will change a dot every 4 frames ( dotDur), so to make the changes as asynchroneous as possible, I change as few dots as possible - that's 3 per frame (setSize)
        setBounds = linspace(1,NdotsToShow+1,dotDur+1); % the sets of dots that will be updated at the same time
        Nsets = length(setBounds)-1; % the number of those sets
        selTemp = NaN(NdotsToShow,1); selStim = [];
        f=1;
        while f <= Nframes+Nsets
          for s = 1:Nsets
            whichDots = setBounds(s):(setBounds(s+1)-1); % which dots go into which set
            selTemp(whichDots) = sampleFrom(setSize,setdiff(1:Ndots,selTemp)); % select a subset of these
            selStim(:,f) = selTemp; % collect
            f = f+1;
          end
        end
        selStim = selStim(:,Nsets:Nframes+Nsets-1); % to chop off the start where there's not yet all dots
        
        % now do the selection again for noise dots
        % difference to stimdots: sample with replacement, spatial offset,
        % random time window.
        % sca; keyboard
        noisex = []; noisez = [];
        if NnoiseDots
          NdotsToShow = NnoiseDots; setSize = NdotsToShow/dotDur; % as framerate is 30Hz, I will change a dot every 4 frames, so to make the changes as asynchroneous as possible, I change as few dots as possible - that's 3 per frame
          setBounds = linspace(1,NdotsToShow+1,dotDur+1);
          Nsets = length(setBounds)-1;
          selTemp = NaN(NdotsToShow,1); sel = [];
          
          offsetx = []; offsetz = []; timeWindow = []; % special for noise dots
          f=1;
          while f <= Nframes+Nsets
            for s = 1:Nsets
              whichDots = setBounds(s):(setBounds(s+1)-1);
              selTemp(whichDots) = sampleFrom(setSize,1:Ndots); % no need to avoid picking the same dot again, as at every sample there's a new spatial offset and time window
              sel(:,f) = selTemp;
              % uniform spatial offsets % CHANGED FROM ORIGINAL: NOT
              % GAUSSIAN, NO NEED FOR IT!
              offsetx(whichDots,f:f+3) = repmat((rand(setSize,1)-.5)*noiseSpread,1,Nsets); % special for noise dots
              offsetz(whichDots,f:f+3) = repmat((rand(setSize,1)-.5)*noiseSpread,1,Nsets); % special for noise dots
              timeStart = ceil(rand(length(whichDots),1)*(Nframes-3)); % special for noise dots
              timeWindow(whichDots,f:f+3) = [timeStart timeStart+1 timeStart+2 timeStart+3]; % special for noise dots
              swapXZ(whichDots,f:f+3) = repmat(round(rand(length(whichDots),1)),1,Nsets);
              flipX(whichDots,f:f+3) = repmat(round(rand(length(whichDots),1))*2-1,1,Nsets); % flip X coordinates or not?
              flipZ(whichDots,f:f+3) = repmat(round(rand(length(whichDots),1))*2-1,1,Nsets); % flip Z coordinates or not?
              f = f+1;
            end
          end
          sel = sel(:,Nsets:Nframes+Nsets-1); % to chop off the start where there's not yet all dots
          offsetx = offsetx(:,Nsets:Nframes+Nsets-1); % to chop off the start where there's not yet all dots
          offsetz = offsetz(:,Nsets:Nframes+Nsets-1); % to chop off the start where there's not yet all dots
          timeWindow = timeWindow(:,Nsets:Nframes+Nsets-1);
          
          % now sample the coordinates of the selected noise dots
          possCoordsNoisex = [ax; bx]; % I REMOVED THE FOLLOWING REST OF THE LINE AS THIS WAS THE CAUSE OF THE WEIRD NOISE DOT CLUSTERINGS: - repmat(mean([ax; bx]')',1,Nframes);
          possCoordsNoisez = [-az; -bz]; % SAME HERE: - repmat(mean([-az; -bz]')',1,Nframes);
          for f=1:Nframes
            for d=1:NnoiseDots
              if swapXZ(d,f)
                noisex(d,f) = flipX(d,f) * possCoordsNoisez(sel(d,f),timeWindow(d,f))+offsetx(sel(d,f));
                noisez(d,f) = flipZ(d,f) * possCoordsNoisex(sel(d,f),timeWindow(d,f))+offsetz(sel(d ,f));
              else
                noisex(d,f) = flipX(d,f) * possCoordsNoisex(sel(d,f),timeWindow(d,f))+offsetx(sel(d,f));
                noisez(d,f) = flipZ(d,f) * possCoordsNoisez(sel(d,f),timeWindow(d,f))+offsetz(sel(d,f));
              end
            end
          end
        end
        
        % now scramble one of the actors if desired
        if showNactors(jj) < 2 % then I scramble at least one
          axMean = mean(ax(:));
          bxMean = mean(bx(:));
          
          possCoordScramActor_ax = ax - axMean;
          possCoordScramActor_az = -az;
          possCoordScramActor_bx = bx - bxMean;
          possCoordScramActor_bz = -bz;
          
          timeWindow = []; % special for noise dots
          f=1;
          NdotsToShow = NstimDots; setSize = NdotsToShow/dotDur; % as framerate is 30Hz, I will change a dot every 4 frames (dotDur), so to make the changes as asynchroneous as possible, I change as few dots as possible - that's 3 per frame (setSize)
          NdotsNeeded = setSize*(dotDur+1);
          temp = ceil(1/(NdotsPerActor / NdotsNeeded)); dotsToSampleFrom = repmat(1:NdotsPerActor,1,temp);
          whichDotsEachSet = reshape(dotsToSampleFrom(1:NdotsNeeded),setSize,dotDur+1)'; % special for scrambled actor
          Nsets = size(whichDotsEachSet,1); % the number of those sets
          selTemp = NaN(NdotsPerActor,1); selScrAct = [];
          while f <= Nframes+Nsets
            for s = 1:Nsets
              whichDots = whichDotsEachSet(s,:);
              selTemp(whichDots) = sampleFrom(setSize,1:NdotsPerActor); % no need to avoid picking the same dot again, as at every sample there's a new spatial offset and time window
              selScrAct(:,f) = selTemp;
              timeStart = ceil(rand(length(whichDots),1)*(Nframes-3)); % special for noise dots
              timeWindow(whichDots,f:f+4) = [timeStart timeStart+1 timeStart+2 timeStart+3 timeStart+3]; % special for noise dots
              %               swapXZ(whichDots,f:f+4) = repmat(zeros(length(whichDots),1),1,5);
              swapXZ(whichDots,f:f+4) = repmat(round(rand(length(whichDots),1)),1,5);
              flipX(whichDots,f:f+4) = repmat(round(rand(length(whichDots),1))*2-1,1,5); % flip X coordinates or not?
              flipZ(whichDots,f:f+4) = repmat(round(rand(length(whichDots),1))*2-1,1,5); % flip Z coordinates or not?
              f = f+1;
            end
          end
          selScrAct = selScrAct(:,Nsets:Nframes+Nsets-1); % to chop off the start where there's not yet all dots
          timeWindow = timeWindow(:,Nsets:Nframes+Nsets-1);
          temp = NaN(size(ax,1),Nframes);
          scramActor_ax = temp; scramActor_az = temp;
          scramActor_bx = temp; scramActor_bz = temp;
          for f=1:Nframes
            for d=1:size(ax,1)
              if swapXZ(d,f)
                scramActor_ax(d,f) = flipX(d,f) * possCoordScramActor_az(selScrAct(d,f),timeWindow(d,f));
                scramActor_az(d,f) = flipZ(d,f) * possCoordScramActor_ax(selScrAct(d,f),timeWindow(d,f));
                scramActor_bx(d,f) = flipX(d,f) * possCoordScramActor_bz(selScrAct(d,f),timeWindow(d,f));
                scramActor_bz(d,f) = flipZ(d,f) * possCoordScramActor_bx(selScrAct(d,f),timeWindow(d,f));
              else
                scramActor_ax(d,f) = flipX(d,f) * possCoordScramActor_ax(selScrAct(d,f),timeWindow(d,f));
                scramActor_az(d,f) = flipZ(d,f) * possCoordScramActor_az(selScrAct(d,f),timeWindow(d,f));
                scramActor_bx(d,f) = flipX(d,f) * possCoordScramActor_bx(selScrAct(d,f),timeWindow(d,f));
                scramActor_bz(d,f) = flipZ(d,f) * possCoordScramActor_bz(selScrAct(d,f),timeWindow(d,f));
              end
            end
          end
          switch showNactors(jj)
            case 0 % scramble both
              scrambleActors(jj,:) = [1 1];
              ax = scramActor_ax + axMean;
              bx = scramActor_bx + bxMean;
              az = scramActor_az;
              bz = scramActor_bz;
            case 1 % scramble 1, the way I did it before
              AorB = rand;
              if AorB<.5 % scramble actor A
                scrambleActors(jj,:) = [1 0];
                ax = scramActor_ax + axMean;
                az = scramActor_az;
              else % scramble actor B
                scrambleActors(jj,:) = [0 1];
                bx = scramActor_bx + bxMean;
                bz = scramActor_bz;
              end
            case 2 % do nothing, this will never be selected anyway
          end % switch showNactors
        else
          scrambleActors(jj,:) = [0 0];
        end % scramble at least 1
        
        % now sample the coordinates of the selected stimulus dots
        tempx = [ax; bx];
        tempz = [-az; -bz];
        
        stimx = []; stimz = [];
        for f = 1:Nframes
          stimx(:,f) = tempx(selStim(:,f),f);
          stimz(:,f) = tempz(selStim(:,f),f);
        end
        
        % swap horizontal locations of actors A and B?
        if swapActorLocations(jj)
          stimx = -stimx;
          stimsAllTrials(jj,:) = stimsAllTrials(jj,[2 1]);
          scrambleActors(jj,:) = scrambleActors(jj,[2 1]);
        end
        
        dotsx = [stimx;noisex];
        dotsz = [stimz;noisez];
        
        % set up the secondary task
        onoff = ceil(Nframes / Nblocks / 2); % determines how many times the squares appear and disappear
        boxCenterPosx = []; boxCenterPosz = []; HorV = [];
        for b = 1:Nblocks;
          boxCenterPosx(b,:) = [-boxx -boxx boxx boxx]+(round(rand(1,4)*boxVar))-(boxVar/2);
          boxCenterPosz(b,:) = [-boxz boxz -boxz boxz]+(round(rand(1,4)*boxVar))-(boxVar/2);
        end
        
        HorV = repmat(shuffle([1 0 0 1]),Nblocks,1);
        if targetSecTask(jj)
          whichSquare = sampleFrom(secondTaskNblocksFlipping,[1:4]);
          startBlockWithTarget = ceil(rand)*(Nblocks-NblocksWithTarget+1); % when does the target start flashing?
          lastBlockWithTarget = NblocksWithTarget+startBlockWithTarget-1; % when does it stop?
          HorV(startBlockWithTarget:2:lastBlockWithTarget,whichSquare) = repmat(HorV(1,whichSquare),NblocksWithTarget/2,1);
          HorV(startBlockWithTarget+1:2:lastBlockWithTarget,whichSquare) = repmat(1-HorV(1,whichSquare),NblocksWithTarget/2,1);
        end
        
        boxCenterPosx = kron(boxCenterPosx,[ones(1,onoff) zeros(1,onoff)]');
        boxCenterPosz = kron(boxCenterPosz,[ones(1,onoff) zeros(1,onoff)]');
        HorV = kron(HorV,[ones(1,onoff) zeros(1,onoff)]');
        
        % now the display
        % initialise observer response variables
        respStart = GetSecs; KeyIsDown = 0; respTime = 0;  KeyCode = zeros(1,256); KbCheck;
        
        % report what will be shown if demo
        if demo
          if congruentInteraction(jj)
            DrawFormattedText(win, ['Jetzt kommt:\n' num2str(indA) ' - ' stimName{indA}], 'center', 'center', 255); %center(1)*.75, center(2)*.3, 255);
          else
            DrawFormattedText(win, 'Jetzt kommt:\n Unpassend', 'center', 'center', 255); %center(1)*.75, center(2)*.3, 255);
          end
          Screen('flip',win);
          pause(1);
          Screen('flip',win);
        end
        
        %% frame loop
        vbl = GetSecs;
        onsets(jj) = GetSecs - startExpt;
        showStim = 1;
%         if strcmpi(whichTask,'fMRItask'); resp(jj) = 0; end % by default it's as if they responded "target absent"
        for i = 1:Nframes;
          boxPosx = repmat(boxCenterPosx(i,:),2,1) + [-1/2 1/2]'*boxLength'*HorV(i,:) + [-1/2 1/2]'*boxWidth'*(1-HorV(i,:)); % make a box around a box center: here the start and end on x axis
          boxPosz = repmat(boxCenterPosz(i,:),2,1) + [-1/2 1/2]'*boxLength'*(1-HorV(i,:)) + [-1/2 1/2]'*boxWidth'*HorV(i,:);
          % get subject response:
          [ KeyIsDown, respTime, KeyCode ] = KbCheck;
          
          % brief check if participant pressed space bar to ask for pause:
          if KeyIsDown && KeyCode(space); pauseFlag(jj) = 1; KeyIsDown = 0; end
          
          if showStim
            % draw the stimulus
            switch dotColor
              case -1
                Screen('DrawDots', win, [dotsx(1:2:end,i), dotsz(1:2:end,i)]'*stimScaling ,dotSize ,0,center,1);
                Screen('DrawDots', win, [dotsx(2:2:end,i), dotsz(2:2:end,i)]'*stimScaling ,dotSize ,255,center,1);
              otherwise
                Screen('DrawDots', win, [dotsx(:,i), dotsz(:,i)]'*stimScaling ,dotSize , dotColor, center,1);
                % playing with Tony with showing some dots in color to see if color spreads to other dots of the same walker - is a walker a perceptual object?
                %                 Screen('DrawDots', win, [dotsx(sel1,i), dotsz(sel1,i)]'*stimScaling ,dotSize , [255 0 0],center,1);
                %                 Screen('DrawDots', win, [dotsx(sel2,i), dotsz(sel2,i)]'*stimScaling ,dotSize , 255,center,1);
            end
            % Ian's Perception 2002 rectangles, make them appear in random non-overlapping locations, and change orientation of one of them. Ian switched them on or off every 400ms and in 50% of trials one of them changed orientation.
            % if secondTask % in Ian's study, the blocks where ALWAYS there, even if participants did not do the secondary task.
            if ~secondTask; HorV = zeros(size(HorV)); end
            if sum(HorV(i,:))~=0
              rect = [boxPosx(1,:);boxPosz(2,:);boxPosx(2,:);boxPosz(1,:)]; % 4 rows per stim: row 1 is left border, 2 is top, 3 is right and 4 is bottom.
              rect = rect * stimScaling; % increase box size
              rect = rect + repmat(kron([1 1],center),4,1)'; % center
              Screen('FillRect', win, 255, rect); % Screen('DrawLines', windowPtr, xy [,width] [,colors] [,center] [,smooth]);
            end
            if ~fMRI
              Screen('DrawText', win, questionMainTask, questionBounds(1), center(2)*1.9, 0);
            end
            
            % get buttonpress during trial
            Screen('DrawingFinished', win); % Tell PTB that no further drawing commands will follow before Screen('Flip')
            vbl = Screen('Flip', win, vbl + (waitScreenRefreshes-0.5)*ifi);
            if KeyIsDown % don't stop the trial if fMRI expt
              if ~fMRI; showStim = 0; end % stops showing stim after button press
              if KeyCode(leftButton)
                resp(jj) = leftResp; RT(jj) = respTime - respStart;
              elseif KeyCode(middleButton)
                resp(jj) = downResp; RT(jj) = respTime - respStart;
              elseif KeyCode(rightButton)
                resp(jj) = rightResp; RT(jj) = respTime - respStart;
              elseif KeyCode(escapeKey)
                sca; ShowCursor; disp('aborted'); if debug; keyboard; else; return; end
                if debug; keyboard; else; return; end
              else
                KeyIsDown = 0;
              end
            end % KeyIsDown
          end % showStim
        end % Nframes
        %% clear screen after frame loop
        offsets(jj) = Screen('Flip',win) - startExpt;  % clear screen

        % get buttonpress after trial
        if ~KeyIsDown && ~demo && (GetSecs - offsets(jj) < noStimResponseTimeWindow) % no answer yet, ask again:
            Screen('DrawText', win, questionMainTask, questionBounds(1), questionBounds(2), 0);
            Screen('Flip',win);  % clear screen
            while ~KeyIsDown
                [ KeyIsDown, respTime, KeyCode ] = KbCheck;
            end
            Screen('Flip',win);  % clear screen
            if KeyCode(leftButton)
                resp(jj) = leftResp; RT(jj) = respTime - respStart;
            elseif KeyCode(middleButton)
                resp(jj) = downResp; RT(jj) = respTime - respStart;
            elseif KeyCode(rightButton)
                resp(jj) = rightResp; RT(jj) = respTime - respStart;
            elseif KeyCode(escapeKey)
                sca; ShowCursor; disp('aborted'); if debug; keyboard; else; return; end
            else
                KeyIsDown = 0;
            end
        end
        
        % Question for 2nd task if required
        if secondTask
          pause(1);
          % show 2nd task question and collect 2nd response
          respStart = GetSecs; KeyIsDown = 0; respTime = 0;  KeyCode = zeros(1,256);
          Screen('DrawText', win, 'No square flipped    |    Square flipped', center(1)*.7, center(2)*1.7, 255);
          Screen('Flip',win);
          while ~KeyIsDown
            [ KeyIsDown, respTime, KeyCode ] = KbCheck;
          end
          Screen('Flip',win);
          if KeyCode(leftButton)
            resp2(jj) = 0; RT2(jj) = respTime - respStart;
          elseif KeyCode(rightButton)
            resp2(jj) = 1; RT2(jj) = respTime - respStart;
          elseif KeyCode(escapeKey)
            sca; ShowCursor; disp('aborted'); if debug; keyboard; else; return; end
          else
            KeyIsDown = 0;
          end
        end
        KeyIsDown = 0;
        
        % the pause crashes PTB! Don't know why.
        %           % now do pause if desired
        %           if ~pauseFlag(jj)
        %             pause(1)
        %             Screen('DrawText', win, '+', center(1), center(2), 255);
        %             Screen('Flip',win);
        %           else
        %             Screen('DrawText', win, 'Pause, weiter mit beliebigem Tastendruck', center(1)*.7, center(2)*1, 255);
        %             Screen('Flip',win);
        %             pause
        %             Screen('DrawText', win, '+', center(1), center(2), 255);
        %             Screen('Flip',win);
        %             pause(1)
        %           end
        % pause(.5)
        
        % assess performance
        if strcmpi(whichTask,'fMRItask')
          if indA == targetFMRI && resp(jj) == targetFMRI; perfCorrect(jj) = 1; end
          if indA == targetFMRI && resp(jj) ~= targetFMRI; perfCorrect(jj) = 0; end
          if indA ~= targetFMRI && resp(jj) == targetFMRI; perfCorrect(jj) = 0; end
          if indA ~= targetFMRI && resp(jj) ~= targetFMRI; perfCorrect(jj) = 1; end
          if showNactors(jj) == 0 && resp(jj) == 0; perfCorrect(jj) = 1; end
        else
          perfCorrect(jj) = 1-abs(targetMain(jj) - resp(jj));
        end
        if ~isnan(targetMain(jj)) && isnan(perfCorrect(jj)); perfCorrect(jj) = 0; end
        if secondTask
          perfCorrectSecTask(jj) = 1-abs(targetSecTask(jj) - resp2(jj));
        end % second task
        
        % report what this was if demo
        if demo
          if congruentInteraction(jj)
            DrawFormattedText(win, ['Das war:\n' num2str(indA) ' - ' stimName{indA}], 'center', 'center', 255); %center(1)*.75, center(2)*.3, 255);
          else
            DrawFormattedText(win, 'Das war:\n Unpassend', 'center', 'center', 255); %center(1)*.75, center(2)*.3, 255);
          end
          Screen('flip',win);
          pause(1.5);
          Screen('flip',win);
          pause(.5);
        end
        % sca; keyboard
        % report performance if desired
        if giveFeedback && ~demo
          if ~isnan(perfCorrect(jj)) % no answer needed -> no feedback
            if perfCorrect(jj) % correct answer
              %               DrawFormattedText(win, ['Richtig, das war(en) ' num2str(targetMain(jj)) ' Person(en)'], 'center', 'center', [0 255 0]) %center(1)*.75, center(2)*.3, 255);
              DrawFormattedText(win, 'Richtig', 'center', 'center', [0 255 0]); %center(1)*.75, center(2)*.3, 255);
            elseif perfCorrect(jj) == 0 % wrong answer
              %               DrawFormattedText(win, ['Falsch, das waren ' num2str(targetMain(jj)) ' Person(en)'], 'center', 'center', [255 0 0]) %center(1)*.75, center(2)*.3, 255);
              DrawFormattedText(win, 'Falsch', 'center', 'center', [255 0 0]); %center(1)*.75, center(2)*.3, 255);
            end
          end
          Screen('flip',win);
          pause(.4);
          Screen('flip',win);
        end
        
        % update adaptive method if needed
        if adaptiveMethod
          % which performance to consider?grrgr
          if ~secondTask; perf = perfCorrect; % consider only perf in main task
          else perf = (perfCorrect+perfCorrectSecTask)/2; end % then consider mean perf across both tasks
          % Update Palamedes structure with response. The better the performance, the more noise dots should be shown
          PM = PAL_AMPM_updatePM(PM,perf(jj));
          PM.response(jj) = perfCorrect(jj);
          PM.NnoiseDotsAll = NnoiseDotsAll;
        end
        
        % ISI
        if giveFeedback; ISI = .3; else; ISI = 1; end % short fixed ISI if feedback, otherwise short fixed one
        if demo; ISI = 0; end % no ISI if demo
        if fMRI; ISI = ISIs(jj); end; % variable ISI if fMRI, overrides all other durations
        Screen('DrawText', win, '+', center(1), center(2), dotColor);
        startISI = Screen('flip',win);
        while GetSecs-startISI < ISI % waits
        end
        Screen('flip',win); % clear screen and move on
        
        data = [[1:jj]' stimsSelected(1:jj,:) showNactors(1:jj) swapActorLocations(1:jj) onsets(1:jj) offsets(1:jj) perfCorrect(1:jj) RT(1:jj)];
        if ~demo; dlmwrite([filenameBase '_dataEachTrial.txt'], data); end
      end % expType
  end % Ntrials
  % Show finishing screen
  DrawFormattedText(win, 'Die Aufgabe ist beendet. Vielen Dank!', 'center', 'center', WhiteIndex(win), [], [], [], lineHeight);
  Screen('Flip', win);
  pause(1);
  sca
catch errorMessage
  sca
  disp('crashed!!')
  errorMessage
   keyboard
end % try Psychtoolbox

%% show results and save data
if ~demo
  % to save:
  if ~secondTask
    data = [[1:jj]' stimsSelected(1:jj,:) showNactors(1:jj) swapActorLocations(1:jj) onsets(1:jj) offsets(1:jj) perfCorrect(1:jj) RT(1:jj)];
  else
    data = [[1:jj]' stimsSelected showNactors congruentInteraction NnoiseDotsAll onsets offsets perfCorrect RT perfCorrectSecTask RT2 pauseFlag];
  end
  dlmwrite([filenameBase '_dataEachTrial.txt'], data);
  
  if adaptiveMethod
    
    % save adaptive procedure data
    PM.LUT = []; % that's too big
    PM.posteriorTplus1givenSuccess = []; % that's too big
    PM.posteriorTplus1givenFailure = []; % that's too big
    
    save([filenameBase '_adaptive.mat'],'stimsAllTrials','congruentInteraction','scrambleActors','perfCorrect','RT','NnoiseDotsAll','onsets','offsets','PM');
    
    % better: show standard output of palamedes toolbox
    try showPMoutput(PM); end
  else % constant stimuli
    if ~fMRI
      [nn,idx] = sort(NnoiseDotsAll);
      pp = reshape(perfCorrect(idx),NtrialsPerNoiselevel,length(NoiseLevels));
      if secondTask;
        pp2 = reshape(perfCorrectSecTask(idx),NtrialsPerNoiselevel,length(NoiseLevels));
        perfEachLevelSecondTask = mean(pp2);
      end
      nn = reshape(nn,NtrialsPerNoiselevel,length(NoiseLevels));
      perfEachLevel = nanmean(pp);
    end
  end
  
end

% try
%   [stimsSelected resp']
%   p_corr_main
%   hit_main
%   FA_main
%   figure;
%   subplot(2,1,1); plot(perfCorrect,'o')
%   subplot(2,1,2); plot(NnoiseDotsAll,'o')
%   mean(NnoiseDotsAll(reversals+4))
% end
% close