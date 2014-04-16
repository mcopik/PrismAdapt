//==============================================================================
//	
//	Copyright (c) 2002-
//	Authors:
//	* Marcin Copik <mcopik@gmail.com> (Silesian University of Technology)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================
package simulator.gpu.opencl.kernel;

import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.accessArrayElement;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.accessStructureField;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.addParentheses;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.convertPrismAction;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.convertPrismGuard;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.convertPrismProperty;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.convertPrismRate;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.createAssignment;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.createBasicExpression;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.createNegation;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.fromString;

import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import parser.ast.ExpressionLiteral;
import simulator.gpu.automaton.AbstractAutomaton;
import simulator.gpu.automaton.AbstractAutomaton.StateVector;
import simulator.gpu.automaton.PrismVariable;
import simulator.gpu.automaton.command.AdaptCommand;
import simulator.gpu.automaton.command.Command;
import simulator.gpu.automaton.command.CommandInterface;
import simulator.gpu.automaton.command.SynchronizedCommand;
import simulator.gpu.automaton.update.Rate;
import simulator.gpu.automaton.update.Update;
import simulator.gpu.opencl.RuntimeConfig;
import simulator.gpu.opencl.kernel.expression.ComplexKernelComponent;
import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.ExpressionGenerator;
import simulator.gpu.opencl.kernel.expression.ExpressionGenerator.Operator;
import simulator.gpu.opencl.kernel.expression.ForLoop;
import simulator.gpu.opencl.kernel.expression.IfElse;
import simulator.gpu.opencl.kernel.expression.KernelComponent;
import simulator.gpu.opencl.kernel.expression.KernelMethod;
import simulator.gpu.opencl.kernel.expression.Method;
import simulator.gpu.opencl.kernel.expression.Switch;
import simulator.gpu.opencl.kernel.memory.ArrayType;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.CLVariable.Location;
import simulator.gpu.opencl.kernel.memory.PointerType;
import simulator.gpu.opencl.kernel.memory.StdVariableType;
import simulator.gpu.opencl.kernel.memory.StdVariableType.StdType;
import simulator.gpu.opencl.kernel.memory.StructureType;
import simulator.sampler.Sampler;
import simulator.sampler.SamplerBoolean;
import simulator.sampler.SamplerBoundedUntilCont;
import simulator.sampler.SamplerBoundedUntilDisc;
import simulator.sampler.SamplerNext;
import simulator.sampler.SamplerUntil;

public abstract class KernelGenerator
{
	protected CLVariable varTime = null;
	protected CLVariable varSelectionSize = null;
	protected CLVariable varStateVector = null;
	protected CLVariable varPathLength = null;
	protected CLVariable varLoopDetection = null;
	protected CLVariable varSynSelectionSize = null;
	protected CLVariable varGuardsTab = null;
	protected CLVariable[] varSynchronizedStates = null;
	protected CLVariable varPropertiesArray = null;

	protected enum KernelMethods {
		/**
		 * DTMC:
		 * Return value is number of concurrent transitions.
		 * int checkGuards(StateVector * sv, bool * guardsTab);
		 * CTMC:
		 * Return value is rates sum of transitions in race condition.
		 * float checkGuards(StateVector * sv, bool * guardsTab);
		 */
		CHECK_GUARDS(0),
		/**
		 * DTMC:
		 * Return value is number of concurrent transitions.
		 * int checkGuardsSyn(StateVector * sv, SynCmdState ** tab);
		 * CTMC:
		 * Return value is rates sum of transitions in race condition.
		 * float checkGuardsSyn(StateVector * sv, SynCmdState * tab);
		 */
		CHECK_GUARDS_SYN(1),
		/**
		 * DTMC:
		 * void performUpdate(StateVector * sv, bool * guardsTab, float sumSelection, int allTransitions);
		 * CTMC:
		 * void performUpdate(StateVector * sv,  bool * guardsTab,float sumSelection);
		 */
		PERFORM_UPDATE(2),
		/**
		 * DTMC:
		 * void performUpdateSyn(StateVector * sv, int updateSelection,SynCmdState * tab);
		 * CTMC:
		 * void performUpdateSyn(StateVector * sv, float sumSelection,SynCmdState * tab);
		 */
		PERFORM_UPDATE_SYN(3),
		/**
		 * Return value determines is we can stop simulation(we know all values).
		 * DTMC:
		 * bool updateProperties(StateVector * sv,PropertyState * prop);
		 * OR
		 * bool updateProperties(StateVector * sv,PropertyState * prop,int time);
		 * 
		 * CTMC:
		 * bool updateProperties(StateVector * sv,PropertyState * prop);
		 * OR
		 * bool updateProperties(StateVector * sv,PropertyState * prop,float time, float updated_time);
		 */
		UPDATE_PROPERTIES(4), CHECK_GUARDS_ADAPT(5), PERFORM_UPDATE_ADAPT(6);
		public final int indice;

		private KernelMethods(int indice)
		{
			this.indice = indice;
		}

		public final static int SIZE = KernelMethods.values().length;
	}

	/**
	 * struct StateVector {
	 * 	each_variable;
	 * }
	 */
	protected StructureType stateVectorType = null;
	public final static StructureType PROPERTY_STATE_STRUCTURE;
	static {
		StructureType type = new StructureType("PropertyState");
		type.addVariable(new CLVariable(new StdVariableType(StdType.BOOL), "propertyState"));
		type.addVariable(new CLVariable(new StdVariableType(StdType.BOOL), "valueKnown"));
		PROPERTY_STATE_STRUCTURE = type;
	}
	protected final static String STATE_VECTOR_PREFIX = "__STATE_VECTOR_";
	protected Map<String, StructureType> synchronizedStates = null;
	protected StructureType synCmdState = null;
	protected AbstractAutomaton model = null;
	protected RuntimeConfig config = null;
	protected Command commands[] = null;
	protected SynchronizedCommand synCommands[] = null;
	protected List<AdaptCommand> adaptCmds = new ArrayList<>();
	protected List<Sampler> properties = null;
	protected List<KernelComponent> additionalDeclarations = new ArrayList<>();
	/**
	 * 
	 */
	protected EnumMap<KernelMethods, Method> helperMethods = new EnumMap<>(KernelMethods.class);
	/**
	 * List of synchronized guard check methods, one for each label.
	 */
	protected List<Method> synchronizedGuards = null;
	/**
	 * List of synchronized update methods, one for each label.
	 */
	protected List<Method> synchronizedUpdates = null;
	/**
	 * List of additional methods.
	 */
	protected List<Method> additionalMethods = null;
	/**
	 * Main kernel method.
	 */
	protected KernelMethod mainMethod = null;
	/**
	 * Pseudo-random number generator type.
	 */
	protected PRNGType prngType = null;
	/**
	 * Variables from state vector.
	 */
	protected PrismVariable[] svVars = null;
	/**
	 * True when model contains synchronized commands.
	 */
	protected boolean hasSynchronized = false;
	/**
	 * True when model contains 'normal' commands.
	 */
	protected boolean hasNonSynchronized = false;
	/**
	 * True when one of processed properties has timing constraints.
	 */
	protected boolean timingProperty = false;

	//	/**
	//	 * Contains names of variables that need to be copied before update.
	//	 */
	//	protected Set<String> nonSynchVarsToSave = new HashSet<>();
	//	/**
	//	 * Contains names of variables that need to be copied before update.
	//	 */
	//	protected Map<String, Set<String>> synchVarsToSave = new HashMap<>();

	/**
	 * 
	 * @param model
	 * @param properties
	 * @param config
	 */
	public KernelGenerator(AbstractAutomaton model, List<Sampler> properties, RuntimeConfig config)
	{
		this.model = model;
		this.properties = properties;
		this.config = config;
		this.prngType = config.prngType;
		this.svVars = model.getStateVector().getVars();
		importStateVector();
		int synSize = model.synchCmdsNumber();
		int size = model.commandsNumber();
		if (synSize != 0) {
			synCommands = new SynchronizedCommand[synSize];
			hasSynchronized = true;
		}
		if (size - synSize != 0) {
			commands = new Command[size - synSize - model.adaptCmdsNumber()];
			hasNonSynchronized = true;
		}
		int normalCounter = 0, synCounter = 0;
		for (int i = 0; i < size; ++i) {
			CommandInterface cmd = model.getCommand(i);
			if (!cmd.isSynchronized()) {
				commands[normalCounter++] = (Command) cmd;
			} else {
				if (cmd instanceof SynchronizedCommand) {
					synCommands[synCounter++] = (SynchronizedCommand) cmd;
				} else {
					AdaptCommand adp = (AdaptCommand) cmd;
					adaptCmds.add(adp);
				}
			}
		}
		for (Sampler sampler : properties) {
			if (sampler instanceof SamplerBoundedUntilCont || sampler instanceof SamplerBoundedUntilDisc) {
				timingProperty = true;
				break;
			}
		}
		if (hasSynchronized) {
			createSynchronizedStructures();
		}
		additionalDeclarations.add(PROPERTY_STATE_STRUCTURE.getDefinition());
		if (prngType.getAdditionalDefinitions() != null) {
			additionalDeclarations.addAll(prngType.getAdditionalDefinitions());
		}
		//		/*
		//		 * copying old values for multiple update - non-synchronized
		//		 */
		//		findVariablesToSave();
		//		if (hasSynchronized) {
		//			for (SynchronizedCommand cmd : synCommands) {
		//				findVariablesToSaveSyn(cmd);
		//			}
		//		}
	}

	//	protected void findUpdatedVariables(Update upd, Set<String> updatedVars, Set<String> varsToSave)
	//	{
	//		for (int i = 0; i < upd.getActionsNumber(); ++i) {
	//			for (Pair<PrismVariable, parser.ast.Expression> pair : upd.getAction(i).expressions) {
	//				//for each action, check whether it contains old variable
	//				for (String entry : updatedVars) {
	//					String updExpr = pair.second.toString();
	//					int index = updExpr.indexOf(entry);
	//					if (index == -1) {
	//						continue;
	//					}
	//					int len = index + entry.length();
	//					//check whether it is not a prefix of longer variable
	//					if ((index + len) != updExpr.length()
	//							&& (Character.isAlphabetic(updExpr.charAt(index + len)) || Character.isDigit(updExpr.charAt(index + len)))) {
	//						continue;
	//					}
	//					//check whether it is not a suffix of longer variable
	//					if (index != 0 && (Character.isAlphabetic(updExpr.charAt(index - 1)) || Character.isDigit(updExpr.charAt(index - 1)))) {
	//						continue;
	//					}
	//					varsToSave.add(entry);
	//				}
	//				updatedVars.add(pair.first.name);
	//			}
	//		}
	//	}
	//
	//	protected void findVariablesToSave()
	//	{
	//		Set<String> updatedVars = new HashSet<>();
	//		if (hasNonSynchronized) {
	//			for (Command cmd : commands) {
	//				Update upd = cmd.getUpdate();
	//				findUpdatedVariables(upd, updatedVars, nonSynchVarsToSave);
	//				updatedVars.clear();
	//			}
	//		}
	//	}
	//
	//	protected void findVariablesToSaveSyn(SynchronizedCommand cmd)
	//	{
	//		Set<String> updatedVars = new HashSet<>();
	//		Update upd = null;
	//		Set<String> varsToSave = new HashSet<>();
	//		for (int i = 0; i < cmd.getModulesNum(); ++i) {
	//			//for each cmd in module
	//			for (int j = 0; j < cmd.getCommandNumber(i); ++j) {
	//				upd = cmd.getCommand(i, j).getUpdate();
	//				findUpdatedVariables(upd, updatedVars, varsToSave);
	//			}
	//		}
	//		synchVarsToSave.put(cmd.synchLabel, varsToSave);
	//	}

	protected abstract void createSynchronizedStructures();

	public StructureType getSVType()
	{
		return stateVectorType;
	}

	public List<KernelComponent> getAdditionalDeclarations()
	{
		if (synchronizedStates != null) {
			for (StructureType type : synchronizedStates.values()) {
				additionalDeclarations.add(type.getDefinition());
			}
		}
		return additionalDeclarations;
	}

	public Collection<Method> getHelperMethods()
	{
		List<Method> ret = new ArrayList<>();
		ret.addAll(helperMethods.values());
		if (synchronizedGuards != null) {
			ret.addAll(synchronizedGuards);
		}
		if (synchronizedUpdates != null) {
			ret.addAll(synchronizedUpdates);
		}
		if (additionalMethods != null) {
			ret.addAll(additionalMethods);
		}
		return ret;
	}

	protected void importStateVector()
	{
		StateVector sv = model.getStateVector();
		stateVectorType = new StructureType("StateVector");
		PrismVariable[] vars = sv.getVars();
		for (int i = 0; i < vars.length; ++i) {
			CLVariable var = new CLVariable(new StdVariableType(vars[i]), translateSVField(vars[i].name));
			stateVectorType.addVariable(var);
		}
		additionalDeclarations.add(stateVectorType.getDefinition());
	}

	protected CLValue initStateVector()
	{
		StateVector sv = model.getStateVector();
		Integer[] init = new Integer[sv.size()];
		if (config.initialState == null) {
			PrismVariable[] vars = sv.getVars();
			for (int i = 0; i < vars.length; ++i) {
				init[i] = vars[i].initValue;
			}
		} else {
			Object[] initVars = config.initialState.varValues;
			for (int i = 0; i < initVars.length; ++i) {
				if (initVars[i] instanceof Integer) {
					init[i] = (Integer) initVars[i];
				} else {
					init[i] = new Integer(((Boolean) initVars[i]) ? 1 : 0);
				}
			}
		}
		return stateVectorType.initializeStdStructure(init);
	}

	/*********************************
	 * MAIN METHOD
	 ********************************/
	public Method createMainMethod() throws KernelException
	{
		Method currentMethod = new KernelMethod();

		/**
		 * Main method arguments.
		 */
		//ARG 0: prng input
		currentMethod.addArg(prngType.getAdditionalInput());
		//ARG 1: number of simulations in this iteration
		CLVariable numberOfSimulations = new CLVariable(new StdVariableType(StdType.UINT32), "numberOfSimulations");
		currentMethod.addArg(numberOfSimulations);
		//ARG 2: sample number for PRNG
		CLVariable sampleNumber = new CLVariable(new StdVariableType(StdType.UINT32), "sampleNumber");
		currentMethod.addArg(sampleNumber);
		//ARG 3: offset in access into global array of results
		CLVariable resultsOffset = new CLVariable(new StdVariableType(StdType.UINT32), "resultsOffset");
		currentMethod.addArg(resultsOffset);
		//ARG 4: offset in access into global array of path
		CLVariable pathOffset = new CLVariable(new StdVariableType(StdType.UINT32), "pathOffset");
		currentMethod.addArg(pathOffset);
		//ARG 4a: probabilities, current
		CLVariable probArray = new CLVariable(new PointerType(new StdVariableType(StdType.FLOAT)), "probArray");
		probArray.setMemoryLocation(Location.GLOBAL);
		currentMethod.addArg(probArray);
		//ARG 5: path lengths buffer
		CLVariable pathLengths = new CLVariable(new PointerType(new StdVariableType(StdType.UINT32)), "pathLengths");
		pathLengths.memLocation = Location.GLOBAL;
		currentMethod.addArg(pathLengths);
		//ARG 6..N: property results
		CLVariable[] propertyResults = new CLVariable[properties.size()];
		for (int i = 0; i < propertyResults.length; ++i) {
			propertyResults[i] = new CLVariable(new PointerType(new StdVariableType(StdType.UINT8)),
			//propertyNumber
					String.format("property%d", i));
			propertyResults[i].memLocation = Location.GLOBAL;
			currentMethod.addArg(propertyResults[i]);
		}
		/**
		 * Local variables.
		 */
		//global ID of thread
		CLVariable globalID = new CLVariable(new StdVariableType(StdType.UINT32), "globalID");
		globalID.setInitValue(ExpressionGenerator.assignGlobalID());
		currentMethod.addLocalVar(globalID);
		//state vector for model
		varStateVector = new CLVariable(stateVectorType, "stateVector");
		varStateVector.setInitValue(initStateVector());
		currentMethod.addLocalVar(varStateVector);
		currentMethod.registerStateVector(varStateVector);

		//property results
		ArrayType propertiesArrayType = new ArrayType(PROPERTY_STATE_STRUCTURE, properties.size());
		varPropertiesArray = new CLVariable(propertiesArrayType, "properties");
		currentMethod.addLocalVar(varPropertiesArray);
		CLValue initValues[] = new CLValue[properties.size()];
		CLValue initValue = PROPERTY_STATE_STRUCTURE.initializeStdStructure(new Number[] { 0, 0 });
		for (int i = 0; i < initValues.length; ++i) {
			initValues[i] = initValue;
		}
		varPropertiesArray.setInitValue(propertiesArrayType.initializeArray(initValues));
		//non-synchronized guard tab
		if (hasNonSynchronized) {
			varGuardsTab = new CLVariable(new ArrayType(new StdVariableType(0, commands.length), commands.length), "guardsTab");
			currentMethod.addLocalVar(varGuardsTab);
		}
		//synchronized state
		if (hasSynchronized) {
			varSynchronizedStates = new CLVariable[synchronizedStates.size()];
			int counter = 0;
			for (Map.Entry<String, StructureType> types : synchronizedStates.entrySet()) {
				varSynchronizedStates[counter] = new CLVariable(types.getValue(),
				//synchState_label
						String.format("synchState_%s", types.getKey()));
				currentMethod.addLocalVar(varSynchronizedStates[counter++]);
			}
		}
		//pathLength
		varPathLength = new CLVariable(new StdVariableType(StdType.UINT32), "pathLength");
		currentMethod.addLocalVar(varPathLength);
		//flag for loop detection
		varLoopDetection = new CLVariable(new StdVariableType(StdType.BOOL), "loopDetection");
		varLoopDetection.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(varLoopDetection);
		//additional local variables, mainly selectionSize. depends on DTMC/CTMC
		mainMethodDefineLocalVars(currentMethod);

		/**
		 * ADAPT: adapt guard number
		 */
		CLVariable varGuardNumber = new CLVariable(new StdVariableType(StdType.INT8), "guardNumber");
		varGuardNumber.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(varGuardNumber);

		/**
		 * Create helpers method.
		 */
		if (hasNonSynchronized) {
			helperMethods.put(KernelMethods.CHECK_GUARDS, createNonsynGuardsMethod());
			helperMethods.put(KernelMethods.PERFORM_UPDATE, createNonsynUpdate());
		}
		if (hasSynchronized) {
			createSynGuardsMethod();
			createUpdateMethodSyn();
		}
		helperMethods.put(KernelMethods.UPDATE_PROPERTIES, createPropertiesMethod());
		helperMethods.put(KernelMethods.CHECK_GUARDS_ADAPT, createAdaptGuardsCheck());

		/**
		 * reject samples with globalID greater than numberOfSimulations
		 */
		currentMethod.addExpression(String.format("if(%s > %s) {\n return;\n}\n", globalID.varName, numberOfSimulations.varName));
		/**
		 * initialize generator
		 */
		currentMethod.addExpression(prngType.initializeGenerator());
		currentMethod.addExpression(prngType.randomize());
		//int size = (int) Math.floor(Math.pow(2.0, adaptCmds.size()));
		//currentMethod.addExpression(createAssignment(mdpAction, config.prngType.getRandomInt(fromString(size)).getSource()));
		mainMethodFirstUpdateProperties(currentMethod);
		ForLoop loop = new ForLoop(varPathLength, (long) 0, config.maxPathLength);
		/**
		 * Check how much numbers are generated with each randomize().
		 * If 1, then we do not need any earlier call - we will randomize variable when we need them.
		 */
		if (prngType.numbersPerRandomize() > 1) {
			//random only when it is necessary
			if (prngType.numbersPerRandomize() != mainMethodRandomsPerIteration()) {
				int iterationsPerRandomize = prngType.numbersPerRandomize() / mainMethodRandomsPerIteration();
				Expression condition = new Expression(String.format("%s %% %d", varPathLength.varName, iterationsPerRandomize));
				IfElse ifElse = new IfElse(createBasicExpression(condition, Operator.EQ, fromString(0)));
				ifElse.addExpression(prngType.randomize());
				loop.addExpression(ifElse);
			}
			//each randomize will give randoms enough for one iteration
			else {
				loop.addExpression(prngType.randomize());
			}
		}
		/**
		 * ADAPT: change the order of conditions
		 */
		loop.addExpression(createAssignment(varGuardNumber, helperMethods.get(KernelMethods.CHECK_GUARDS_ADAPT).callMethod(varStateVector.convertToPointer())));
		IfElse ifElse = new IfElse(createBasicExpression(varGuardNumber.getSource(), Operator.EQ, fromString(-1)));
		/**
		 * check which guards are active
		 */
		if (hasNonSynchronized) {
			Expression callCheckGuards = helperMethods.get(KernelMethods.CHECK_GUARDS).callMethod(
			//(stateVector,guardsTab)
					varStateVector.convertToPointer(), varGuardsTab);
			ifElse.addExpression(createAssignment(varSelectionSize, callCheckGuards));
		}
		if (hasSynchronized) {
			ifElse.addExpression(createAssignment(varSynSelectionSize, fromString(0)));
			for (int i = 0; i < synCommands.length; ++i) {
				Expression callMethod = synchronizedGuards.get(i).callMethod(
				//&stateVector
						varStateVector.convertToPointer(),
						//synchState
						varSynchronizedStates[i].convertToPointer());
				ifElse.addExpression(createBasicExpression(varSynSelectionSize.getSource(), Operator.ADD_AUGM, callMethod));
			}
		}
		/**
		 * if(selectionSize + synSelectionSize == 0) -> deadlock, break
		 */
		Expression sum = null;
		if (varSynSelectionSize != null && varSelectionSize != null) {
			sum = createBasicExpression(varSelectionSize.getSource(), Operator.ADD, varSynSelectionSize.getSource());
		} else if (varSynSelectionSize != null) {
			sum = varSynSelectionSize.getSource();
		} else {
			sum = varSelectionSize.getSource();
		}
		IfElse deadlockState = new IfElse(createBasicExpression(sum, Operator.EQ, fromString(0)));
		mainMethodUpdateProperties(deadlockState);
		deadlockState.addExpression(new Expression("break;\n"));
		ifElse.addExpression(deadlockState);
		loop.addExpression(ifElse);
		//l//oop.addExpression(new Expression("if(globalID<5)printf(\"selection %d %d %d \\n\",globalID,pathLength,stateVector.__STATE_VECTOR_phase);"));

		//		loop.addExpression(new Expression(
		//				"if(globalID<5)printf(\"selection gID %d %d %d %d %d %d %d %d\\n\",globalID,stateVector.__STATE_VECTOR_x1,stateVector.__STATE_VECTOR_x2,stateVector.__STATE_VECTOR_x3,stateVector.__STATE_VECTOR_x4,stateVector.__STATE_VECTOR_x5,stateVector.__STATE_VECTOR_x6,stateVector.__STATE_VECTOR_x7);"));
		/**
		 * update time -> in case of CTMC and bounded until we need two time values:
		 * 1) entering state
		 * 2) leaving state
		 * so current time is updated in After method()
		 * other cases: compute time in Before method() 
		 */
		mainMethodUpdateTimeBefore(currentMethod, loop);
		//		loop.addExpression(new Expression(
		//				"if(stateVector.__STATE_VECTOR_tosses > 95 && stateVector.__STATE_VECTOR_head == 50 /*&& pathLength < 20*/)printf(\"end loop gID %d %d %d %d %d %d\\n\",globalID,pathLength,stateVector.__STATE_VECTOR_s,stateVector.__STATE_VECTOR_tosses,stateVector.__STATE_VECTOR_head,stateVector.__STATE_VECTOR_adapt_toss_diff);"));
		/**
		 * if all properties are known, then we can end iterating
		 */
		mainMethodUpdateProperties(loop);
		//loop.addExpression(new Expression("if(get_global_id(0) < 10)printf(\"%d %f %f \",get_global_id(0),time,updatedTime);\n"));
		IfElse callAdaptUpdate = new IfElse(createBasicExpression(varGuardNumber.getSource(), Operator.EQ, fromString(-1)));
		/**
		 * call update method; 
		 * most complex case - both nonsyn and synchronized updates
		 */
		if (hasNonSynchronized && hasSynchronized) {
			mainMethodCallBothUpdates(callAdaptUpdate);
		}
		/**
		 * only synchronized updates
		 */
		else if (hasSynchronized) {
			mainMethodCallSynUpdate(callAdaptUpdate);
		}
		/**
		 * only nonsyn updates
		 */
		else {
			mainMethodCallNonsynUpdate(callAdaptUpdate);
		}
		callAdaptUpdate.addElse();
		/**
		 * ADAPT: for each adapt command
		 */
		CLVariable rnd = new CLVariable(new StdVariableType(StdType.FLOAT), "rnd");
		rnd.setInitValue(config.prngType.getRandomUnifFloat(new Expression(String.format("%s%%%d", varPathLength.getSource().toString(),
				config.prngType.numbersPerRandomize()))));
		callAdaptUpdate.addExpression(1, rnd.getDefinition());
		Switch _switch = new Switch(varGuardNumber);
		int count = 0, accumulatedSum = 0;
		Map<String, String> translations = new HashMap<>();
		for (CLVariable var : stateVectorType.getFields()) {
			translations.put(var.varName, translateSVField(var.varName));
		}
		for (AdaptCommand adp : adaptCmds) {
			List<String> updates = adp.getNames();
			_switch.addCase(fromString(count));
			Expression tabPosition = null;
			if (adp.var != null) {
				tabPosition = varStateVector.accessField(translateSVField(adp.var.name)).getSource();
				tabPosition = createBasicExpression(tabPosition, Operator.ADD, fromString(accumulatedSum - adp.var.minValue));
			} else {
				tabPosition = fromString(accumulatedSum);
			}
			CLVariable prob = probArray.accessElement(tabPosition);
			IfElse binaryCondition = new IfElse(createBasicExpression(rnd.getSource(), Operator.LT, prob.getSource()));
			binaryCondition.addExpression(createBasicExpression(rnd.getSource(), Operator.DIV_AUGM, prob.getSource()));
			createUpdates(binaryCondition, translations, svVars, rnd, adp.getUpdate(updates.get(0)));
			binaryCondition.addElse();
			binaryCondition.setConditionNumber(1);
			// (rnd - prob) / (1-prob)
			binaryCondition.addExpression(createAssignment(
					rnd,
					createBasicExpression(addParentheses(createBasicExpression(rnd.getSource(), Operator.SUB, prob.getSource())), Operator.DIV,
							addParentheses(createBasicExpression(fromString(1), Operator.SUB, prob.getSource())))));

			createUpdates(binaryCondition, translations, svVars, rnd, adp.getUpdate(updates.get(1)));
			_switch.addExpression(count, binaryCondition);
			accumulatedSum += adp.getBranchNum();
			++count;
		}
		callAdaptUpdate.addExpression(1, createAssignment(varLoopDetection, fromString("false")));
		callAdaptUpdate.addExpression(1, _switch);
		loop.addExpression(callAdaptUpdate);
		/**
		 * For CTMC&bounded until -> update current time.
		 */
		mainMethodUpdateTimeAfter(currentMethod, loop);
		//		loop.addExpression(new Expression(
		//				"if(stateVector.__STATE_VECTOR_tosses > 95 && stateVector.__STATE_VECTOR_head == 50 /*&& pathLength < 20*/)printf(\"end loop2 gID %d %d %d %d %d %d\\n\",globalID,pathLength,stateVector.__STATE_VECTOR_s,stateVector.__STATE_VECTOR_tosses,stateVector.__STATE_VECTOR_head,stateVector.__STATE_VECTOR_adapt_toss_diff);"));

		mainMethodLoopDetection(loop);
		//		loop.addExpression(new Expression(
		//				"if(stateVector.__STATE_VECTOR_tosses > 95 && stateVector.__STATE_VECTOR_head == 50 /*&& pathLength < 20*/)printf(\"end loop3 gID %d %d %d %d %d %d\\n\",globalID,pathLength,stateVector.__STATE_VECTOR_s,stateVector.__STATE_VECTOR_tosses,stateVector.__STATE_VECTOR_head,stateVector.__STATE_VECTOR_adapt_toss_diff);"));

		//loop.addExpression(new Expression("if(phase==4)break;"));
		currentMethod.addExpression(loop);
		/**
		 * Write results.
		 */
		//sampleNumber + globalID
		Expression position = createBasicExpression(globalID.getSource(), Operator.ADD, pathOffset.getSource());
		//path length
		CLVariable pathLength = pathLengths.varType.accessElement(pathLengths, position);
		currentMethod.addExpression(createAssignment(pathLength, varPathLength));
		//		currentMethod
		//				.addExpression(new Expression(
		//						"if(pathLength ==0)printf(\"%d %d %d\\n\",stateVector.__STATE_VECTOR_adapt_toss_ratio,stateVector.__STATE_VECTOR_s0,stateVector.__STATE_VECTOR_tosses)"));
		position = createBasicExpression(globalID.getSource(), Operator.ADD, resultsOffset.getSource());
		//each property result
		for (int i = 0; i < properties.size(); ++i) {
			CLVariable result = accessArrayElement(propertyResults[i], position);
			CLVariable property = accessArrayElement(varPropertiesArray, fromString(i)).accessField("propertyState");
			currentMethod.addExpression(createAssignment(result, property));
		}
		currentMethod.addExpression(prngType.deinitializeGenerator());

		currentMethod.addInclude(prngType.getIncludes());
		return currentMethod;
	}

	void createUpdates(ComplexKernelComponent root, Map<String, String> translations, PrismVariable[] vars, CLVariable rnd, Update update)
	{
		Rate rate = new Rate(update.getRate(0));
		IfElse ifElse = new IfElse(createBasicExpression(rnd.getSource(), Operator.LT, fromString(convertPrismRate(vars, rate))));
		if (!update.isActionTrue(0)) {
			//			if (!timingProperty) {
			//				ifElse.addExpression(0, convertPrismAction(update.getAction(0), translations, changeFlag, oldValue));
			//			} else {
			ifElse.addExpression(0, convertPrismAction(update.getAction(0), translations));
			//			}
		}
		for (int j = 1; j < update.getActionsNumber(); ++j) {
			rate.addRate(update.getRate(j));
			ifElse.addElif(createBasicExpression(rnd.getSource(), Operator.LT, fromString(convertPrismRate(vars, rate))));
			if (!update.isActionTrue(j)) {
				//				if (!timingProperty) {
				//					ifElse.addExpression(j, convertPrismAction(update.getAction(j), translations, changeFlag, oldValue));
				//				} else {
				ifElse.addExpression(j, convertPrismAction(update.getAction(j), translations));
				//				}
			}
		}
		root.addExpression(ifElse);
	}

	protected abstract int mainMethodRandomsPerIteration();

	protected abstract void mainMethodDefineLocalVars(Method currentMethod) throws KernelException;

	protected void mainMethodCallBothUpdates(ComplexKernelComponent parent)
	{
		//selection
		Expression sum = createBasicExpression(varSelectionSize.getSource(), Operator.ADD, varSynSelectionSize.getSource());
		addParentheses(sum);
		CLVariable selection = mainMethodSelectionVar(sum);
		parent.addExpression(selection.getDefinition());
		//		parent.addExpression(new Expression(
		//				"if(get_global_id(0) < 5)printf(\"%d %f %d %d %d\\n\",get_global_id(0),selection,stateVector.__STATE_VECTOR_q,stateVector.__STATE_VECTOR_s,stateVector.__STATE_VECTOR_s2);\n"));

		IfElse ifElse = mainMethodBothUpdatesCondition(selection);
		/**
		 * else
		 * callSynUpdate()
		 */
		ifElse.addElse();
		//write everything to else
		ifElse.setConditionNumber(1);
		CLVariable synSum = mainMethodBothUpdatesSumVar();
		//start checking values from the low limit of non-syn size
		synSum.setInitValue(varSelectionSize);
		ifElse.addExpression(synSum.getDefinition());
		mainMethodCallSynUpdate(ifElse, selection, synSum, sum);
		parent.addExpression(ifElse);
	}

	protected void mainMethodCallSynUpdate(ComplexKernelComponent parent)
	{
		CLVariable selection = mainMethodSelectionVar(varSynSelectionSize.getSource());
		CLVariable synSum = mainMethodBothUpdatesSumVar();
		synSum.setInitValue(StdVariableType.initialize(0));
		parent.addExpression(selection.getDefinition());
		parent.addExpression(synSum.getDefinition());
		mainMethodCallSynUpdate(parent, selection, synSum, varSynSelectionSize.getSource());
	}

	protected void mainMethodCallSynUpdate(ComplexKernelComponent parent, CLVariable selection, CLVariable synSum, Expression sum)
	{
		if (synCommands.length > 1) {
			/**
			 * Loop counter, over all labels
			 */
			CLVariable counter = new CLVariable(new StdVariableType(0, synCommands.length), "synSelection");
			counter.setInitValue(StdVariableType.initialize(0));
			parent.addExpression(counter.getDefinition());
			ForLoop loop = new ForLoop(counter, 0, synCommands.length);
			Switch _switch = new Switch(counter);
			for (int i = 0; i < synCommands.length; ++i) {
				CLVariable currentSize = varSynchronizedStates[i].accessField("size");
				_switch.addCase(fromString(i));
				_switch.addExpression(i, createBasicExpression(synSum.getSource(), Operator.ADD_AUGM,
				// synSum += synchState__label.size;
						currentSize.getSource()));
			}
			loop.addExpression(_switch);
			/**
			 * Check whether we have found proper label.
			 */
			IfElse checkSelection = new IfElse(mainMethodSynUpdateCondition(selection, synSum, sum));
			/**
			 * If yes, then counter shows us the label.
			 * For each one, recompute probability/rate
			 */
			_switch = new Switch(counter);
			for (int i = 0; i < synCommands.length; ++i) {
				_switch.addCase(fromString(i));
				_switch.setConditionNumber(i);
				/**
				 * Remove current size, added in loop.
				 */
				CLVariable currentSize = varSynchronizedStates[i].accessField("size");
				_switch.addExpression(i, createBasicExpression(synSum.getSource(), Operator.SUB_AUGM,
				// synSum -= synchState__label.size;
						currentSize.getSource()));
				/**
				 * Recompute probability/rate
				 */
				mainMethodSynRecomputeSelection(_switch, selection, synSum, sum, currentSize);
			}
			checkSelection.addExpression(_switch);
			checkSelection.addExpression("break;\n");
			loop.addExpression(checkSelection);
			parent.addExpression(loop);
			/**
			 * Counter shows selected label, so we can call the update.
			 */
			_switch = new Switch(counter);
			for (int i = 0; i < synCommands.length; ++i) {
				_switch.addCase(fromString(i));
				Expression call = synchronizedUpdates.get(i).callMethod(
				//&stateVector
						varStateVector.convertToPointer(),
						//&synchState__label
						varSynchronizedStates[i].convertToPointer(),
						//probability
						selection);
				_switch.addExpression(i, timingProperty ? call : createAssignment(varLoopDetection, call));
			}
			parent.addExpression(_switch);
		} else {
			CLVariable currentSize = varSynchronizedStates[0].accessField("size");
			/**
			 * Recompute probability/rate
			 */
			mainMethodSynRecomputeSelection(parent, selection, synSum, sum, currentSize);
			Expression call = synchronizedUpdates.get(0).callMethod(
			//&stateVector
					varStateVector.convertToPointer(),
					//&synchState__label
					varSynchronizedStates[0].convertToPointer(),
					//probability
					selection);
			parent.addExpression(timingProperty ? call : createAssignment(varLoopDetection, call));
		}
	}

	protected void mainMethodLoopDetection(ComplexKernelComponent parent)
	{
		if (!timingProperty) {
			Expression updateFlag = createBasicExpression(varLoopDetection.getSource(), Operator.EQ, fromString("true"));

			Expression updateSize = null;
			if (hasNonSynchronized && hasSynchronized) {
				updateSize = createBasicExpression(varSelectionSize.getSource(), Operator.ADD, varSynSelectionSize.getSource());
			} else if (hasNonSynchronized) {
				updateSize = varSelectionSize.getSource();
			} else {
				updateSize = varSynSelectionSize.getSource();
			}

			updateSize = createBasicExpression(updateSize, Operator.EQ, fromString("1"));
			IfElse loop = new IfElse(createBasicExpression(updateFlag, Operator.LAND, updateSize));
			loop.setConditionNumber(0);
			mainMethodUpdateProperties(loop);
			loop.addExpression(new Expression("break;\n"));
			parent.addExpression(loop);
		}
	}

	protected abstract CLVariable mainMethodBothUpdatesSumVar();

	protected abstract IfElse mainMethodBothUpdatesCondition(CLVariable selection);

	protected abstract Expression mainMethodSynUpdateCondition(CLVariable selection, CLVariable synSum, Expression sum);

	protected abstract void mainMethodSynRecomputeSelection(ComplexKernelComponent parent, CLVariable selection, CLVariable synSum, Expression sum,
			CLVariable currentLabelSize);

	protected abstract CLVariable mainMethodSelectionVar(Expression selectionSize);

	protected abstract void mainMethodCallNonsynUpdate(ComplexKernelComponent parent);

	protected abstract void mainMethodFirstUpdateProperties(ComplexKernelComponent parent);

	protected abstract void mainMethodUpdateProperties(ComplexKernelComponent currentMethod);

	protected abstract void mainMethodUpdateTimeBefore(Method currentMethod, ComplexKernelComponent parent);

	protected abstract void mainMethodUpdateTimeAfter(Method currentMethod, ComplexKernelComponent parent);

	/*********************************
	 * NON-SYNCHRONIZED GUARDS CHECK
	 ********************************/
	protected Method createNonsynGuardsMethod() throws KernelException
	{
		if (!hasNonSynchronized) {
			return null;
		}
		Method currentMethod = guardsMethodCreateSignature();
		//StateVector * sv
		CLVariable sv = new CLVariable(varStateVector.getPointer(), "sv");
		currentMethod.addArg(sv);
		currentMethod.registerStateVector(sv);
		//bool * guardsTab
		CLVariable guards = new CLVariable(varGuardsTab.getPointer(), "guardsTab");
		currentMethod.addArg(guards);
		//counter
		CLVariable counter = new CLVariable(new StdVariableType(0, commands.length), "counter");
		counter.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(counter);
		guardsMethodCreateLocalVars(currentMethod);
		for (int i = 0; i < commands.length; ++i) {
			guardsMethodCreateCondition(currentMethod, i, convertPrismGuard(svVars, commands[i].getGuard().toString()));
		}
		//TODO: do I need this?
		//signature last guard
		CLVariable position = guards.varType.accessElement(guards, new Expression(counter.varName));
		IfElse ifElse = new IfElse(createBasicExpression(counter.getSource(), Operator.NE, fromString(commands.length)));
		ifElse.addExpression(0, createAssignment(position, fromString(commands.length)));
		currentMethod.addExpression(ifElse);
		guardsMethodReturnValue(currentMethod);
		return currentMethod;
	}

	protected abstract Method guardsMethodCreateSignature();

	protected abstract void guardsMethodCreateLocalVars(Method currentMethod) throws KernelException;

	protected abstract void guardsMethodCreateCondition(Method currentMethod, int position, String guard);

	protected abstract void guardsMethodReturnValue(Method currentMethod);

	protected Method createAdaptGuardsCheck() throws KernelException
	{
		Method currentMethod = new Method("checkAdaptGuards", new StdVariableType(StdType.INT8));
		//StateVector * sv
		CLVariable sv = new CLVariable(varStateVector.getPointer(), "sv");
		currentMethod.addArg(sv);
		currentMethod.registerStateVector(sv);
		int i = 0;
		for (AdaptCommand cmd : adaptCmds) {
			IfElse ifElse = new IfElse(new Expression(convertPrismGuard(svVars, cmd.guard.toString())));
			ifElse.addExpression(new Expression(String.format("return %d;", i)));
			currentMethod.addExpression(ifElse);
		}
		currentMethod.addReturn(fromString(-1));
		//		//counter
		//		CLVariable counter = new CLVariable(new StdVariableType(0, commands.length), "counter");
		//		counter.setInitValue(StdVariableType.initialize(0));
		//		currentMethod.addLocalVar(counter);
		//		guardsMethodCreateLocalVars(currentMethod);
		//		for (int i = 0; i < commands.length; ++i) {
		//			guardsMethodCreateCondition(currentMethod, i, convertPrismGuard(svVars, commands[i].getGuard().toString()));
		//		}
		//		//TODO: do I need this?
		//		//signature last guard
		//		CLVariable position = guards.varType.accessElement(guards, new Expression(counter.varName));
		//		IfElse ifElse = new IfElse(createBasicExpression(counter.getSource(), Operator.NE, fromString(commands.length)));
		//		ifElse.addExpression(0, createAssignment(position, fromString(commands.length)));
		//		currentMethod.addExpression(ifElse);
		//		guardsMethodReturnValue(currentMethod);
		return currentMethod;
	}

	/*********************************
	 * NON-SYNCHRONIZED UPDATE
	 ********************************/
	protected Method createNonsynUpdate() throws KernelException
	{
		if (!hasNonSynchronized) {
			return null;
		}
		Method currentMethod = new Method("updateNonsynGuards", new StdVariableType(timingProperty ? StdType.VOID : StdType.BOOL));
		//StateVector * sv
		CLVariable sv = new CLVariable(varStateVector.getPointer(), "sv");
		currentMethod.addArg(sv);
		currentMethod.registerStateVector(sv);
		//bool * guardsTab
		CLVariable guards = new CLVariable(new PointerType(new StdVariableType(0, commands.length)), "guardsTab");
		currentMethod.addArg(guards);
		//float sum
		CLVariable selectionSum = new CLVariable(new StdVariableType(StdType.FLOAT), "selectionSum");
		selectionSum.setInitValue(StdVariableType.initialize(0.0f));
		currentMethod.addArg(selectionSum);
		CLVariable selection = new CLVariable(new StdVariableType(0, commands.length), "selection");
		selection.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(selection);
		//changeFlag
		CLVariable changeFlag = new CLVariable(new StdVariableType(StdType.BOOL), "changeFlag");
		changeFlag.setInitValue(StdVariableType.initialize(1));
		currentMethod.addLocalVar(changeFlag);
		//oldValue
		CLVariable oldValue = new CLVariable(new StdVariableType(StdType.INT32), "oldValue");
		oldValue.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(oldValue);

		updateMethodAdditionalArgs(currentMethod);
		updateMethodLocalVars(currentMethod);
		updateMethodPerformSelection(currentMethod);
		CLVariable guardsTabSelection = accessArrayElement(varGuardsTab, selection.getSource());
		Switch _switch = new Switch(guardsTabSelection.getSource());
		int switchCounter = 0;
		PrismVariable[] vars = model.getStateVector().getVars();
		Map<String, String> translations = new HashMap<>();
		for (PrismVariable var : vars) {
			translations.put(var.name, String.format("((float)%s)", var.name));
		}
		for (int i = 0; i < commands.length; ++i) {
			Update update = commands[i].getUpdate();
			Rate rate = new Rate(update.getRate(0));
			if (update.getActionsNumber() > 1) {
				IfElse ifElse = new IfElse(createBasicExpression(selectionSum.getSource(), Operator.LT, fromString(convertPrismRate(vars, rate))));
				if (!update.isActionTrue(0)) {
					if (!timingProperty) {
						ifElse.addExpression(0, convertPrismAction(update.getAction(0), translations, changeFlag, oldValue));
					} else {
						ifElse.addExpression(0, convertPrismAction(update.getAction(0), translations));
					}
				}
				for (int j = 1; j < update.getActionsNumber(); ++j) {
					rate.addRate(update.getRate(j));
					ifElse.addElif(createBasicExpression(selectionSum.getSource(), Operator.LT, fromString(convertPrismRate(vars, rate))));
					if (!update.isActionTrue(j)) {
						if (!timingProperty) {
							ifElse.addExpression(j, convertPrismAction(update.getAction(j), translations, changeFlag, oldValue));
						} else {
							ifElse.addExpression(j, convertPrismAction(update.getAction(j), translations));
						}
					}
				}
				_switch.addCase(new Expression(Integer.toString(i)));
				_switch.addExpression(switchCounter++, ifElse);
			} else {
				if (!update.isActionTrue(0)) {
					_switch.addCase(new Expression(Integer.toString(i)));
					if (!timingProperty) {
						_switch.addExpression(switchCounter++, convertPrismAction(update.getAction(0), translations, changeFlag, oldValue));
					} else {
						_switch.addExpression(switchCounter++, convertPrismAction(update.getAction(0), translations));
					}
				}
			}
		}
		currentMethod.addExpression(_switch);
		if (!timingProperty) {
			currentMethod.addReturn(changeFlag);
		}
		return currentMethod;
	}

	protected abstract void updateMethodPerformSelection(Method currentMethod) throws KernelException;

	protected abstract void updateMethodAdditionalArgs(Method currentMethod) throws KernelException;

	protected abstract void updateMethodLocalVars(Method currentMethod) throws KernelException;

	/*********************************
	 * PROPERTY METHODS
	 ********************************/
	/**
	 * 
	 * @return
	 * @throws KernelException
	 */
	protected Method createPropertiesMethod() throws KernelException
	{
		Method currentMethod = new Method("checkProperties", new StdVariableType(StdType.BOOL));
		/**
		 * Local variables and args.
		 */
		//StateVector * sv
		CLVariable sv = new CLVariable(varStateVector.getPointer(), "sv");
		currentMethod.addArg(sv);
		currentMethod.registerStateVector(sv);
		//PropertyState * property
		CLVariable propertyState = new CLVariable(new PointerType(PROPERTY_STATE_STRUCTURE), "propertyState");
		currentMethod.addArg(propertyState);
		propertiesMethodTimeArg(currentMethod);
		//uint counter
		CLVariable counter = new CLVariable(new StdVariableType(0, properties.size()), "counter");
		counter.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(counter);
		//bool allKnown
		CLVariable allKnown = new CLVariable(new StdVariableType(StdType.BOOL), "allKnown");
		allKnown.setInitValue(StdVariableType.initialize(1));
		currentMethod.addLocalVar(allKnown);
		/**
		 * For each property, add checking
		 */
		for (int i = 0; i < properties.size(); ++i) {
			Sampler property = properties.get(i);
			CLVariable currentProperty = accessArrayElement(propertyState, counter.getSource());
			CLVariable valueKnown = currentProperty.accessField("valueKnown");
			/**
			 * It should have been done earlier.
			 */
			//			if (!(property instanceof SamplerBoolean)) {
			//				throw new KernelException("Currently rewards are not supported!");
			//			}
			IfElse ifElse = new IfElse(createNegation(valueKnown.getSource()));
			ifElse.addExpression(0, createAssignment(allKnown, fromString("false")));
			/**
			 * X state_formulae
			 * I don't think that this will be used in future.
			 */
			if (property instanceof SamplerNext) {
				propertiesMethodAddNext(ifElse, (SamplerNext) property, currentProperty);
			}
			/**
			 * state_formulae U state_formulae
			 */
			else if (property instanceof SamplerUntil) {
				propertiesMethodAddUntil(ifElse, (SamplerUntil) property, currentProperty);
			}
			/**
			 * state_formulae U[k1,k2] state_formulae
			 * Requires additional timing args.
			 */
			else {
				propertiesMethodAddBoundedUntil(currentMethod, ifElse, (SamplerBoolean) property, currentProperty);
			}
			ifElse.addExpression(0, createAssignment(allKnown, valueKnown));
			currentMethod.addExpression(ifElse);
		}
		currentMethod.addReturn(allKnown);
		return currentMethod;
	}

	protected abstract void propertiesMethodTimeArg(Method currentMethod) throws KernelException;

	protected abstract void propertiesMethodAddBoundedUntil(Method currentMethod, ComplexKernelComponent parent, SamplerBoolean property, CLVariable propertyVar);

	protected void propertiesMethodAddNext(ComplexKernelComponent parent, SamplerNext property, CLVariable propertyVar)
	{
		IfElse ifElse = createPropertyCondition(propertyVar, false, property.getExpression().toString(), true);
		createPropertyCondition(ifElse, propertyVar, false, null, false);
		parent.addExpression(ifElse);
	}

	protected void propertiesMethodAddUntil(ComplexKernelComponent parent, SamplerUntil property, CLVariable propertyVar)
	{
		IfElse ifElse = createPropertyCondition(propertyVar, false, property.getRightSide().toString(), true);
		/**
		 * in F/G it is true
		 */
		if (!(property.getLeftSide() instanceof ExpressionLiteral)) {
			createPropertyCondition(ifElse, propertyVar, true, property.getLeftSide().toString(), false);
		}
		parent.addExpression(ifElse);
	}

	protected IfElse createPropertyCondition(CLVariable propertyVar, boolean negation, String condition, boolean propertyValue)
	{
		IfElse ifElse = null;
		if (!negation) {
			ifElse = new IfElse(convertPrismProperty(svVars, condition));
		} else {
			ifElse = new IfElse(createNegation(convertPrismProperty(svVars, condition)));
		}
		CLVariable valueKnown = accessStructureField(propertyVar, "valueKnown");
		CLVariable propertyState = accessStructureField(propertyVar, "propertyState");
		if (propertyValue) {
			ifElse.addExpression(0, createAssignment(propertyState, fromString("true")));
		} else {
			ifElse.addExpression(0, createAssignment(propertyState, fromString("false")));
		}
		ifElse.addExpression(0, createAssignment(valueKnown, fromString("true")));
		return ifElse;
	}

	protected void createPropertyCondition(IfElse ifElse, CLVariable propertyVar, boolean negation, String condition, boolean propertyValue)
	{
		if (condition != null) {
			if (!negation) {
				ifElse.addElif(convertPrismProperty(svVars, condition));
			} else {
				ifElse.addElif(createNegation(convertPrismProperty(svVars, condition)));
			}
		} else {
			ifElse.addElse();
		}
		CLVariable valueKnown = accessStructureField(propertyVar, "valueKnown");
		CLVariable propertyState = accessStructureField(propertyVar, "propertyState");
		if (propertyValue) {
			ifElse.addExpression(ifElse.size() - 1, createAssignment(propertyState, fromString("true")));
		} else {
			ifElse.addExpression(ifElse.size() - 1, createAssignment(propertyState, fromString("false")));
		}
		ifElse.addExpression(ifElse.size() - 1, createAssignment(valueKnown, fromString("true")));
	}

	/*********************************
	 * SYNCHRONIZED GUARDS CHECK
	 ********************************/
	protected void createSynGuardsMethod()
	{
		if (!hasSynchronized) {
			return;
		}
		synchronizedGuards = new ArrayList<>();
		for (SynchronizedCommand cmd : synCommands) {
			//synchronized state
			CLVariable synState = new CLVariable(new PointerType(synchronizedStates.get(cmd.synchLabel)), "synState");
			int max = cmd.getMaxCommandsNum();
			Method current = guardsSynCreateMethod(String.format("guardCheckSyn__%s", cmd.synchLabel), max);
			CLVariable stateVector = new CLVariable(varStateVector.getPointer(), "sv");
			//size for whole label
			CLVariable labelSize = guardsSynLabelVar(max);
			labelSize.setInitValue(StdVariableType.initialize(0));
			//size for current module
			CLVariable currentSize = guardsSynCurrentVar(max);
			currentSize.setInitValue(StdVariableType.initialize(0));
			CLVariable saveSize = synState.accessField("moduleSize");
			CLVariable guardsTab = synState.accessField("guards");
			try {
				current.addArg(stateVector);
				current.addArg(synState);
				current.addLocalVar(labelSize);
				current.addLocalVar(currentSize);
			} catch (KernelException e) {
				throw new RuntimeException(e);
			}
			current.registerStateVector(stateVector);
			int guardCounter = 0;
			//first module
			for (int i = 0; i < cmd.getCommandNumber(0); ++i) {
				guardsSynAddGuard(current, guardsTab.accessElement(fromString(guardCounter++)),
				//guardsTab[counter] = evaluate(guard)
						cmd.getCommand(0, i), currentSize);
			}
			current.addExpression(createAssignment(saveSize.accessElement(fromString(0)), currentSize));
			current.addExpression(createAssignment(labelSize, currentSize));
			//rest
			for (int i = 1; i < cmd.getModulesNum(); ++i) {
				IfElse ifElse = new IfElse(createBasicExpression(labelSize.getSource(), Operator.NE, fromString(0)));
				ifElse.addExpression(createAssignment(currentSize, fromString(0)));
				for (int j = 0; j < cmd.getCommandNumber(i); ++j) {
					guardsSynAddGuard(ifElse, guardsTab.accessElement(fromString(guardCounter++)),
					//guardsTab[counter] = evaluate(guard)
							cmd.getCommand(i, j), currentSize);
				}
				ifElse.addExpression(createBasicExpression(labelSize.getSource(),
				// cmds_for_label *= cmds_for_module;
						Operator.MUL_AUGM, currentSize.getSource()));
				ifElse.addExpression(createAssignment(saveSize.accessElement(fromString(i)), currentSize));
				current.addExpression(ifElse);
			}
			saveSize = synState.accessField("size");
			current.addExpression(createAssignment(saveSize, labelSize));
			current.addReturn(labelSize);
			synchronizedGuards.add(current);
		}
	}

	protected abstract Method guardsSynCreateMethod(String label, int maxCommandsNumber);

	protected abstract CLVariable guardsSynLabelVar(int maxCommandsNumber);

	protected abstract CLVariable guardsSynCurrentVar(int maxCommandsNumber);

	protected abstract void guardsSynAddGuard(ComplexKernelComponent parent, CLVariable guardArray, Command cmd, CLVariable size);

	/*********************************
	 * SYNCHRONIZED UPDATE
	 ********************************/

	protected void createUpdateMethodSyn()
	{
		//		for (Map.Entry<String, Set<String>> entry : synchVarsToSave.entrySet()) {
		//			for (String var : entry.getValue())
		//				System.out.println(entry.getKey() + " " + var);
		//		}
		synchronizedUpdates = new ArrayList<>();
		additionalMethods = new ArrayList<>();
		for (SynchronizedCommand cmd : synCommands) {

			/**
			 * create updateSynchronized__Label method
			 * takes three args:
			 * - state vector
			 * - module number
			 * - selected guard
			 * - pointer to probability(updates it)
			 */
			Method update = updateSynLabelMethod(cmd);
			additionalMethods.add(update);

			Method current = new Method(String.format("updateSyn__%s", cmd.synchLabel), new StdVariableType(timingProperty ? StdType.VOID : StdType.BOOL));
			//state vector
			CLVariable stateVector = new CLVariable(varStateVector.getPointer(), "sv");
			//synchronized state
			CLVariable synState = new CLVariable(new PointerType(synchronizedStates.get(cmd.synchLabel)), "synState");
			CLVariable propability = new CLVariable(new StdVariableType(StdType.FLOAT), "prop");
			//current guard
			CLVariable guard = new CLVariable(new StdVariableType(0, cmd.getMaxCommandsNum()), "guard");
			guard.setInitValue(StdVariableType.initialize(0));
			//sv->size
			CLVariable saveSize = synState.accessField("size");
			//sv->guards
			CLVariable guardsTab = synState.accessField("guards");
			//size for current module
			CLVariable totalSize = new CLVariable(saveSize.varType, "totalSize");
			totalSize.setInitValue(saveSize);
			//size for current module
			CLVariable oldSV = new CLVariable(stateVectorType, "oldSV");
			oldSV.setInitValue(stateVector.dereference());
			//changeFlag - for loop detection
			CLVariable changeFlag = null;
			if (!timingProperty) {
				changeFlag = new CLVariable(new StdVariableType(StdType.BOOL), "changeFlag");
				changeFlag.setInitValue(StdVariableType.initialize(1));
			}
			try {
				current.addArg(stateVector);
				current.addArg(synState);
				current.addArg(propability);
				current.addLocalVar(oldSV);
				current.addLocalVar(guard);
				current.addLocalVar(totalSize);
				current.registerStateVector(stateVector);
				if (!timingProperty) {
					current.addLocalVar(changeFlag);
				}
				updateSynAdditionalVars(current, cmd);
			} catch (KernelException e) {
				throw new RuntimeException(e);
			}
			current.registerStateVector(stateVector);
			//current.addExpression(new Expression("if(get_global_id(0)<5)printf(\"" + cmd.synchLabel + " %f\\n\",prop);"));
			CLVariable moduleSize = null;
			//			ForLoop loop = new ForLoop("loopCounter", 0, cmd.getModulesNum());
			//			CLVariable loopCounter = loop.getLoopCounter();
			//for-each module
			//TODO: check optimizing without loop unrolling?
			for (int i = 0; i < cmd.getModulesNum(); ++i) {

				moduleSize = synState.accessField("moduleSize").accessElement(fromString(i));
				updateSynBeforeUpdateLabel(current, cmd, i, guardsTab, guard, moduleSize, totalSize, propability);
				/**
				 * call selected update
				 */
				//				current.addExpression(new Expression("if(get_global_id(0) < 10)printf(\"%d " + cmd.synchLabel
				//						+ " %f %d %f %f\\n\",get_global_id(0),prop,guard,totalSize,totalSize * (*synState).moduleSize[" + i + "]);\n"));
				Expression callUpdate = update.callMethod(stateVector, oldSV.convertToPointer(), guardsTab, StdVariableType.initialize(i), guard,
						propability.convertToPointer());
				if (timingProperty) {
					current.addExpression(callUpdate);
				} else {
					current.addExpression(createAssignment(changeFlag, createBasicExpression(callUpdate, Operator.LAND, changeFlag.getSource())));
				}

				updateSynAfterUpdateLabel(current, guard, moduleSize, totalSize, propability);
			}
			//current.addExpression(new Expression("if(get_global_id(0)<5)printf(\"" + cmd.synchLabel + " %f %d %d\\n\",prop,guard,(*sv).__STATE_VECTOR_q);"));

			//			//for-each module
			//			for (int i = 0; i < cmd.getModulesNum(); ++i) {
			//				//moduleSize = fromString(cmd.getCommandNumber(i));
			//				/**
			//				 * compute current guard in update
			//				 */
			//				guardUpdate = functionCall("floor",
			//				//probability * module_size
			//						createBasicExpression(propability.getSource(), Operator.MUL, moduleSize.getSource()));
			//				current.addExpression(createAssignment(guard, guardUpdate));
			//				/**
			//				 * recompute probability to an [0,1) in selected guard
			//				 */
			//				probUpdate = createBasicExpression(
			//				//probability * module_size
			//						createBasicExpression(propability.getSource(), Operator.MUL, moduleSize.getSource()), Operator.SUB,
			//						//guard
			//						guard.getSource());
			//				current.addExpression(createAssignment(propability, probUpdate));
			//				//current.addExpression(new Expression("if(get_global_id(0)<5)printf(\"" + cmd.synchLabel + " %f %d %d\\n\",prop,guard,(*sv).__STATE_VECTOR_q);"));
			//				/**
			//				 * call selected update
			//				 */
			//				current.addExpression(update.callMethod(stateVector, oldSV.convertToPointer(), guardsTab, StdVariableType.initialize(i), guard,
			//						propability.convertToPointer()));
			//
			//				//current.addExpression(new Expression("if(get_global_id(0)<5)printf(\"" + cmd.synchLabel + " %f %d %d\\n\",prop,guard,(*sv).__STATE_VECTOR_q);"));
			//				/**
			//				 * totalSize /= 1 is useless
			//				 */
			//				current.addExpression(createBasicExpression(totalSize.getSource(), Operator.DIV_AUGM, moduleSize.getSource()));
			//			}
			if (!timingProperty) {
				current.addReturn(changeFlag);
			}
			synchronizedUpdates.add(current);
		}
	}

	protected abstract void updateSynAdditionalVars(Method parent, SynchronizedCommand cmd) throws KernelException;

	protected abstract void updateSynBeforeUpdateLabel(Method parent, SynchronizedCommand cmd, int moduleNumber, CLVariable guardsTab, CLVariable guard,
			CLVariable moduleSize, CLVariable totalSize, CLVariable probability);

	protected abstract void updateSynAfterUpdateLabel(ComplexKernelComponent parent, CLVariable guard, CLVariable moduleSize, CLVariable totalSize,
			CLVariable probability);

	protected Method updateSynLabelMethod(SynchronizedCommand synCmd)
	{
		Method current = new Method(String.format("updateSynchronized__%s", synCmd.synchLabel),
		//don't return anything
				new StdVariableType(timingProperty ? StdType.VOID : StdType.BOOL));
		CLVariable stateVector = new CLVariable(varStateVector.getPointer(), "sv");
		CLVariable oldSV = new CLVariable(varStateVector.getPointer(), "oldSV");
		//guardsTab
		CLVariable guardsTab = new CLVariable(new PointerType(new StdVariableType(StdType.BOOL)), "guards");
		//selected module
		CLVariable module = new CLVariable(new StdVariableType(StdType.UINT8), "module");
		CLVariable guard = new CLVariable(new StdVariableType(StdType.UINT8), "guard");
		CLVariable probabilityPtr = new CLVariable(new PointerType(new StdVariableType(StdType.FLOAT)), "prob");
		CLVariable probability = probabilityPtr.dereference();
		CLVariable newValue = null, changeFlag = null;
		if (!timingProperty) {
			newValue = new CLVariable(new StdVariableType(StdType.INT32), "newValue");
			newValue.setInitValue(StdVariableType.initialize(0));
			changeFlag = new CLVariable(new StdVariableType(StdType.BOOL), "changeFlag");
			changeFlag.setInitValue(StdVariableType.initialize(0));
		}
		try {
			current.addArg(stateVector);
			current.addArg(oldSV);
			current.addArg(guardsTab);
			current.addArg(module);
			current.addArg(guard);
			current.addArg(probabilityPtr);
			current.registerStateVector(stateVector);
			if (!timingProperty) {
				current.addLocalVar(newValue);
				current.addLocalVar(changeFlag);
			}
		} catch (KernelException e) {
			throw new RuntimeException(e);
		}
		Map<String, String> translations = new HashMap<>();
		for (CLVariable var : stateVectorType.getFields()) {
			String name = var.varName.substring(STATE_VECTOR_PREFIX.length());
			CLVariable second = oldSV.accessField(var.varName);
			translations.put(name, second.varName);
		}
		Switch _switch = new Switch(module);
		Update update = null;
		Rate rate = null;
		Command cmd = null;
		int moduleOffset = 0;
		CLVariable guardSelection = updateSynLabelMethodGuardSelection(synCmd, guard);
		CLVariable guardCounter = updateSynLabelMethodGuardCounter(synCmd);
		if (guardCounter != null) {
			current.addExpression(guardCounter.getDefinition());
		}
		current.addExpression(guardSelection.getDefinition());
		//for-each module
		for (int i = 0; i < synCmd.getModulesNum(); ++i) {
			_switch.addCase(fromString(i));
			//_switch.addExpression(i, new Expression("if(get_global_id(0)<5)printf(\"" + synCmd.synchLabel + " %d %d %f\\n\",module,guard,*prob);"));
			//_switch.addCommand(i, new Expression("if(get_global_id(0)<5)printf(\"" + synCmd.synchLabel
			//		+ " %d %d %d %d\\n\",(*sv).__STATE_VECTOR_x1 ? 1 : 0 ,(*sv).__STATE_VECTOR_x2,(*sv).__STATE_VECTOR_x3,(*sv).__STATE_VECTOR_x4);"));

			_switch.setConditionNumber(i);
			updateSynLabelMethodSelectGuard(current, _switch, guardSelection, guardCounter, moduleOffset);

			Switch internalSwitch = new Switch(guardSelection);
			//for-each command
			for (int j = 0; j < synCmd.getCommandNumber(i); ++j) {
				cmd = synCmd.getCommand(i, j);
				update = cmd.getUpdate();
				rate = new Rate(update.getRate(0));

				internalSwitch.addCase(fromString(j));
				//when update is in form prob:action + prob:action + ...
				if (update.getActionsNumber() > 1) {
					IfElse ifElse = new IfElse(createBasicExpression(probability.getSource(), Operator.LT, fromString(convertPrismRate(svVars, rate))));
					if (!update.isActionTrue(0)) {
						ifElse.addExpression(0, updateSynProbabilityRecompute(probability, null, rate));
						//ifElse.addExpression(0, convertPrismAction(update.getAction(0)));
						if (!timingProperty) {
							ifElse.addExpression(0, convertPrismAction(update.getAction(0), translations, changeFlag, newValue));
						} else {
							ifElse.addExpression(0, convertPrismAction(update.getAction(0), translations));
						}
					}
					for (int k = 1; k < update.getActionsNumber(); ++k) {
						Rate previous = new Rate(rate);
						rate.addRate(update.getRate(k));
						ifElse.addElif(createBasicExpression(probability.getSource(), Operator.LT, fromString(convertPrismRate(svVars, rate))));
						ifElse.addExpression(k, updateSynProbabilityRecompute(probability, previous, update.getRate(k)));
						if (!update.isActionTrue(k)) {
							//ifElse.addExpression(k, convertPrismAction(update.getAction(k)));
							if (!timingProperty) {
								ifElse.addExpression(k, convertPrismAction(update.getAction(k), translations, changeFlag, newValue));
							} else {
								ifElse.addExpression(k, convertPrismAction(update.getAction(k), translations));
							}
						}
					}
					internalSwitch.addExpression(j, ifElse);
				} else {
					if (!update.isActionTrue(0)) {
						//internalSwitch.addExpression(j, convertPrismAction(update.getAction(0)));
						if (!timingProperty) {
							internalSwitch.addExpression(j, convertPrismAction(update.getAction(0), translations, changeFlag, newValue));
						} else {
							internalSwitch.addExpression(j, convertPrismAction(update.getAction(0), translations));
						}
						//no recomputation necessary!
					}
				}
			}
			moduleOffset += synCmd.getCommandNumber(i);
			_switch.addExpression(i, internalSwitch);
		}
		current.addExpression(_switch);
		if (!timingProperty) {
			current.addReturn(changeFlag);
		}
		return current;
	}

	protected abstract CLVariable updateSynLabelMethodGuardSelection(SynchronizedCommand cmd, CLVariable guard);

	protected abstract CLVariable updateSynLabelMethodGuardCounter(SynchronizedCommand cmd);

	protected abstract void updateSynLabelMethodSelectGuard(Method currentMethod, ComplexKernelComponent parent, CLVariable guardSelection,
			CLVariable guardCounter, int moduleOffset);

	protected abstract Expression updateSynProbabilityRecompute(CLVariable probability, Rate before, Rate current);

	/*********************************
	 * OTHER METHODS
	 ********************************/
	public String translateSVField(String varName)
	{
		return String.format("%s%s", STATE_VECTOR_PREFIX, varName);
	}

}