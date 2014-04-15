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

import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.addComma;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.addParentheses;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.convertPrismGuard;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.convertPrismRate;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.createAssignment;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.createBasicExpression;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.fromString;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.functionCall;
import static simulator.gpu.opencl.kernel.expression.ExpressionGenerator.postIncrement;

import java.util.HashMap;
import java.util.List;

import parser.ast.ExpressionLiteral;
import prism.Preconditions;
import simulator.gpu.automaton.AbstractAutomaton;
import simulator.gpu.automaton.command.Command;
import simulator.gpu.automaton.command.SynchronizedCommand;
import simulator.gpu.automaton.update.Rate;
import simulator.gpu.opencl.RuntimeConfig;
import simulator.gpu.opencl.kernel.expression.ComplexKernelComponent;
import simulator.gpu.opencl.kernel.expression.Expression;
import simulator.gpu.opencl.kernel.expression.ExpressionGenerator;
import simulator.gpu.opencl.kernel.expression.ExpressionGenerator.Operator;
import simulator.gpu.opencl.kernel.expression.ForLoop;
import simulator.gpu.opencl.kernel.expression.IfElse;
import simulator.gpu.opencl.kernel.expression.Method;
import simulator.gpu.opencl.kernel.memory.ArrayType;
import simulator.gpu.opencl.kernel.memory.CLValue;
import simulator.gpu.opencl.kernel.memory.CLVariable;
import simulator.gpu.opencl.kernel.memory.RValue;
import simulator.gpu.opencl.kernel.memory.StdVariableType;
import simulator.gpu.opencl.kernel.memory.StdVariableType.StdType;
import simulator.gpu.opencl.kernel.memory.StructureType;
import simulator.sampler.Sampler;
import simulator.sampler.SamplerBoolean;
import simulator.sampler.SamplerBoundedUntilDisc;

public class KernelGeneratorDTMC extends KernelGenerator
{
	protected int maximalNumberOfSynchsUpdates = 0;

	public KernelGeneratorDTMC(AbstractAutomaton model, List<Sampler> properties, RuntimeConfig config)
	{
		super(model, properties, config);
	}

	@Override
	protected void createSynchronizedStructures()
	{
		synchronizedStates = new HashMap<>();
		CLVariable size = null, array = null, guards = null;
		int sum = 1, max = 0;
		int cmdNumber = 0;
		for (SynchronizedCommand cmd : synCommands) {
			StructureType type = new StructureType(String.format("SynState__%s", cmd.synchLabel));

			for (int i = 0; i < cmd.getModulesNum(); ++i) {
				cmdNumber = cmd.getCommandNumber(i);
				if (cmdNumber > max) {
					max = cmdNumber;
				}
			}
			sum = cmd.getMaxCommandsNum();
			maximalNumberOfSynchsUpdates += sum;
			size = new CLVariable(new StdVariableType(0, sum), "size");
			array = new CLVariable(new ArrayType(new StdVariableType(0, max), cmd.getModulesNum()), "moduleSize");
			guards = new CLVariable(new ArrayType(new StdVariableType(StdType.BOOL), cmd.getCmdsNum()), "guards");
			type.addVariable(size);
			type.addVariable(array);
			type.addVariable(guards);
			synchronizedStates.put(cmd.synchLabel, type);
		}
	}

	/*********************************
	 * MAIN METHOD
	 ********************************/
	@Override
	protected void mainMethodFirstUpdateProperties(ComplexKernelComponent parent)
	{
		//in case of DTMC, there is nothing to do
	}

	@Override
	public void mainMethodDefineLocalVars(Method currentMethod) throws KernelException
	{
		//time
		varTime = new CLVariable(new StdVariableType(0, config.maxPathLength), "time");
		varTime.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(varTime);
		//number of transitions
		varSelectionSize = new CLVariable(new StdVariableType(0, model.commandsNumber()), "selectionSize");
		varSelectionSize.setInitValue(StdVariableType.initialize(0));
		currentMethod.addLocalVar(varSelectionSize);
		//number of synchronized transitions
		if (hasSynchronized) {
			for (SynchronizedCommand cmd : synCommands) {
				maximalNumberOfSynchsUpdates += cmd.getMaxCommandsNum();
			}
			varSynSelectionSize = new CLVariable(new StdVariableType(0, maximalNumberOfSynchsUpdates - 1), "selectionSynSize");
			varSynSelectionSize.setInitValue(StdVariableType.initialize(0));
			currentMethod.addLocalVar(varSynSelectionSize);
		}
	}

	@Override
	protected void mainMethodUpdateTimeBefore(Method currentMethod, ComplexKernelComponent parent)
	{
		parent.addExpression(addComma(postIncrement(currentMethod.getLocalVar("time"))));
	}

	@Override
	protected void mainMethodUpdateTimeAfter(Method currentMethod, ComplexKernelComponent parent)
	{
		//don't need to do anything!
	}

	@Override
	protected int mainMethodRandomsPerIteration()
	{
		return 1;
	}

	//
	//	protected void mainMethodCallBothUpdates(ComplexKernelComponent parent)
	//	{
	//		//selection
	//		CLVariable selection = new CLVariable(new StdVariableType(StdType.FLOAT), "selection");
	//		Expression sum = createBasicExpression(varSelectionSize.getSource(), Operator.ADD, varSynSelectionSize.getSource());
	//		addParentheses(sum);
	//
	//		Expression rndNumber = new Expression(String.format("%s%%%d", varPathLength.getSource().toString(), config.prngType.numbersPerRandomize()));
	//		selection.setInitValue(config.prngType.getRandomUnifFloat(rndNumber));
	//		parent.addExpression(selection.getDefinition());
	//		Expression condition = createBasicExpression(selection.getSource(), Operator.LT,
	//		//nonSyn/(syn+nonSyn)
	//				createBasicExpression(varSelectionSize.cast("float"), Operator.DIV, sum));
	//		IfElse ifElse = new IfElse(condition);
	//		/**
	//		 * if(selection < selectionSize/sum)
	//		 * callNonsynUpdate(..)
	//		 */
	//		ifElse.addExpression(mainMethodCallNonsynUpdate(new RValue(rndNumber), new RValue(sum)));
	//		/**
	//		 * else
	//		 * callSynUpdate()
	//		 */
	//		//TODO: add case for one
	//		ifElse.addElse();
	//		CLVariable counter = new CLVariable(new StdVariableType(0, synCommands.length), "synSelection");
	//		counter.setInitValue(StdVariableType.initialize(0));
	//		CLVariable synSum = new CLVariable(new StdVariableType(0, maximalNumberOfSynchsUpdates), "synSum");
	//		synSum.setInitValue(varSelectionSize);
	//		ifElse.addExpression(1, counter.getDefinition());
	//		ifElse.addExpression(1, synSum.getDefinition());
	//		ForLoop loop = new ForLoop(counter, 0, synCommands.length);
	//		Switch _switch = new Switch(counter);
	//		for (int i = 0; i < synCommands.length; ++i) {
	//			CLVariable currentSize = varSynchronizedStates[i].accessField("size");
	//			_switch.addCase(fromString(i));
	//			_switch.addExpression(i, createBasicExpression(synSum.getSource(), Operator.ADD_AUGM,
	//			// synSum += synchState__label.size;
	//					currentSize.getSource()));
	//		}
	//		loop.addExpression(_switch);
	//		IfElse checkSelection = new IfElse(createBasicExpression(selection.getSource(), Operator.LT,
	//		//probability < synSum/sum
	//				createBasicExpression(synSum.cast("float"), Operator.DIV, sum)));
	//		_switch = new Switch(counter);
	//		Expression probUpdate = createBasicExpression(selection.getSource(), Operator.MUL, sum);
	//		probUpdate = createBasicExpression(probUpdate, Operator.SUB, synSum.getSource());
	//		for (int i = 0; i < synCommands.length; ++i) {
	//			CLVariable currentSize = varSynchronizedStates[i].accessField("size");
	//			_switch.addCase(fromString(i));
	//			_switch.addExpression(i, createBasicExpression(synSum.getSource(), Operator.SUB_AUGM,
	//			// synSum += synchState__label.size;
	//					currentSize.getSource()));
	//			_switch.addExpression(i, createAssignment(selection, probUpdate));
	//			_switch.addExpression(i, createBasicExpression(selection.getSource(), Operator.DIV_AUGM, currentSize.getSource()));
	//		}
	//		checkSelection.addExpression(_switch);
	//		checkSelection.addExpression("break;\n");
	//		loop.addExpression(checkSelection);
	//		ifElse.addExpression(1, loop);
	//		_switch = new Switch(counter);
	//		for (int i = 0; i < synCommands.length; ++i) {
	//			_switch.addCase(fromString(i));
	//			_switch.addExpression(i, synchronizedUpdates.get(i).callMethod(
	//			//&stateVector
	//					varStateVector.convertToPointer(),
	//					//&synchState__label
	//					varSynchronizedStates[i].convertToPointer(),
	//					//probability
	//					selection));
	//		}
	//		ifElse.addExpression(1, _switch);
	//
	//		parent.addExpression(ifElse);
	//		//		parent.addExpression(new Expression(
	//		//				"if(globalID<5)printf(\"%d %d %d %d\\n\",globalID,stateVector.__STATE_VECTOR_q,stateVector.__STATE_VECTOR_s,stateVector.__STATE_VECTOR_s2);"));
	//	}
	//
	//	protected void mainMethodCallSynUpdate(ComplexKernelComponent parent)
	//	{
	//		CLVariable selection = new CLVariable(new StdVariableType(StdType.FLOAT), "selection");
	//		Expression rndNumber = new Expression(String.format("%s%%%d", varPathLength.getSource().toString(), config.prngType.numbersPerRandomize()));
	//		selection.setInitValue(config.prngType.getRandomUnifFloat(rndNumber));
	//		parent.addExpression(selection.getDefinition());
	//		//parent.addExpression(new Expression("if(globalID<5)printf(\"selection %d %f\\n\",globalID,selection);"));
	//		CLVariable counter = new CLVariable(new StdVariableType(0, synCommands.length), "synSelection");
	//		counter.setInitValue(StdVariableType.initialize(0));
	//		CLVariable synSum = new CLVariable(new StdVariableType(0, maximalNumberOfSynchsUpdates), "synSum");
	//		synSum.setInitValue(StdVariableType.initialize(0));
	//		parent.addExpression(counter.getDefinition());
	//		parent.addExpression(synSum.getDefinition());
	//		ForLoop loop = new ForLoop(counter, 0, synCommands.length);
	//		Switch _switch = new Switch(counter);
	//		for (int i = 0; i < synCommands.length; ++i) {
	//			CLVariable currentSize = varSynchronizedStates[i].accessField("size");
	//			_switch.addCase(fromString(i));
	//			_switch.addExpression(i, createBasicExpression(synSum.getSource(), Operator.ADD_AUGM,
	//			// synSum += synchState__label.size;
	//					currentSize.getSource()));
	//		}
	//		loop.addExpression(_switch);
	//		IfElse checkSelection = new IfElse(createBasicExpression(selection.getSource(), Operator.LT,
	//		//probability < synSum/sum
	//				createBasicExpression(synSum.cast("float"), Operator.DIV, varSynSelectionSize.getSource())));
	//		_switch = new Switch(counter);
	//		Expression probUpdate = createBasicExpression(selection.getSource(), Operator.MUL, varSynSelectionSize.getSource());
	//		probUpdate = createBasicExpression(probUpdate, Operator.SUB, synSum.getSource());
	//		for (int i = 0; i < synCommands.length; ++i) {
	//			CLVariable currentSize = varSynchronizedStates[i].accessField("size");
	//			_switch.addCase(fromString(i));
	//			_switch.addExpression(i, createBasicExpression(synSum.getSource(), Operator.SUB_AUGM,
	//			// synSum += synchState__label.size;
	//					currentSize.getSource()));
	//			_switch.addExpression(i, createAssignment(selection, probUpdate));
	//			_switch.addExpression(i, createBasicExpression(selection.getSource(), Operator.DIV_AUGM, currentSize.getSource()));
	//		}
	//		checkSelection.addExpression(_switch);
	//		checkSelection.addExpression("break;\n");
	//		loop.addExpression(checkSelection);
	//		parent.addExpression(loop);
	//		_switch = new Switch(counter);
	//		for (int i = 0; i < synCommands.length; ++i) {
	//			_switch.addCase(fromString(i));
	//			_switch.addExpression(i, synchronizedUpdates.get(i).callMethod(
	//			//&stateVector
	//					varStateVector.convertToPointer(),
	//					//&synchState__label
	//					varSynchronizedStates[i].convertToPointer(),
	//					//probability
	//					selection));
	//		}
	//		parent.addExpression(_switch);
	//	}

	@Override
	protected void mainMethodCallNonsynUpdate(ComplexKernelComponent parent)
	{
		Expression rndNumber = new Expression(String.format("%s%%%d", varPathLength.getSource().toString(), config.prngType.numbersPerRandomize()));
		CLValue random = config.prngType.getRandomUnifFloat(rndNumber);
		parent.addExpression(mainMethodCallNonsynUpdate(random, varSelectionSize));
	}

	private Expression mainMethodCallNonsynUpdate(CLValue rnd, CLValue sum)
	{
		Method update = helperMethods.get(KernelMethods.PERFORM_UPDATE);
		Expression call = update.callMethod(
		//stateVector
				varStateVector.convertToPointer(),
				//non-synchronized guards tab
				varGuardsTab,
				//random float [0,1]
				rnd,
				//number of commands
				sum);
		return timingProperty ? call : createAssignment(varLoopDetection, call);
	}

	@Override
	protected CLVariable mainMethodBothUpdatesSumVar()
	{
		return new CLVariable(new StdVariableType(0, maximalNumberOfSynchsUpdates), "synSum");
	}

	@Override
	protected IfElse mainMethodBothUpdatesCondition(CLVariable selection)
	{
		Expression sum = createBasicExpression(varSelectionSize.getSource(), Operator.ADD, varSynSelectionSize.getSource());
		addParentheses(sum);
		Expression condition = createBasicExpression(selection.getSource(), Operator.LT,
		//nonSyn/(syn+nonSyn)
				createBasicExpression(varSelectionSize.cast("float"), Operator.DIV, sum));
		IfElse ifElse = new IfElse(condition);
		/**
		 * if(selection < selectionSize/sum)
		 * callNonsynUpdate(..)
		 */
		ifElse.addExpression(mainMethodCallNonsynUpdate(selection, new RValue(sum)));
		return ifElse;
	}

	@Override
	protected Expression mainMethodSynUpdateCondition(CLVariable selection, CLVariable synSum, Expression sum)
	{
		return createBasicExpression(selection.getSource(), Operator.LT,
		//probability < synSum/sum
				createBasicExpression(synSum.cast("float"), Operator.DIV, sum));
	}

	@Override
	protected void mainMethodSynRecomputeSelection(ComplexKernelComponent parent, CLVariable selection, CLVariable synSum, Expression sum,
			CLVariable currentLabelSize)
	{
		//selection*sum - synSum
		Expression probUpdate = createBasicExpression(selection.getSource(), Operator.MUL, sum);
		probUpdate = createBasicExpression(probUpdate, Operator.SUB, synSum.getSource());
		/**
		 * Takes value from [synSum/sum,synSum/sum+currentLabelSize/sum] to an interval [0,1]
		 * selection = (selection*sum -synSum)/currentLabelSize
		 */
		parent.addExpression(createAssignment(selection, probUpdate));
		parent.addExpression(createBasicExpression(selection.getSource(), Operator.DIV_AUGM, currentLabelSize.getSource()));
	}

	@Override
	protected CLVariable mainMethodSelectionVar(Expression selectionSize)
	{
		CLVariable selection = new CLVariable(new StdVariableType(StdType.FLOAT), "selection");
		Expression rndNumber = new Expression(String.format("%s%%%d", varPathLength.getSource().toString(),
		//pathLength%2 for Random123
				config.prngType.numbersPerRandomize()));
		selection.setInitValue(config.prngType.getRandomUnifFloat(fromString(rndNumber)));
		return selection;
	}

	@Override
	protected void mainMethodUpdateProperties(ComplexKernelComponent parent)
	{
		Expression call = null;
		if (timingProperty) {
			call = helperMethods.get(KernelMethods.UPDATE_PROPERTIES).callMethod(varStateVector.convertToPointer(), varPropertiesArray, varTime);
		} else {
			call = helperMethods.get(KernelMethods.UPDATE_PROPERTIES).callMethod(varStateVector.convertToPointer(), varPropertiesArray);
		}
		IfElse ifElse = new IfElse(call);
		//		ifElse.addExpression(
		//				0,
		//				new Expression(
		//						"if(get_global_id(0) < 10)printf(\"%f %f %f %d %d\\n\",time,updatedTime,selectionSize,stateVector.__STATE_VECTOR_q,properties[0].propertyState);\n"));
		ifElse.addExpression(0, new Expression("break;\n"));
		parent.addExpression(ifElse);
	}

	/*********************************
	 * NON-SYNCHRONIZED GUARDS CHECK
	 ********************************/
	@Override
	protected void guardsMethodCreateLocalVars(Method currentMethod) throws KernelException
	{
		//don't need to do anything!
	}

	@Override
	protected Method guardsMethodCreateSignature()
	{
		return new Method("checkNonsynGuards", new StdVariableType(0, commands.length - 1));
	}

	@Override
	protected void guardsMethodCreateCondition(Method currentMethod, int position, String guard)
	{
		CLVariable guardsTab = currentMethod.getArg("guardsTab");
		Preconditions.checkNotNull(guardsTab, "");
		CLVariable counter = currentMethod.getLocalVar("counter");
		Preconditions.checkNotNull(counter, "");
		CLVariable tabPos = guardsTab.varType.accessElement(guardsTab, ExpressionGenerator.postIncrement(counter));
		IfElse ifElse = new IfElse(new Expression(guard));
		ifElse.addExpression(0, createAssignment(tabPos, fromString(position)));
		currentMethod.addExpression(ifElse);
	}

	@Override
	protected void guardsMethodReturnValue(Method currentMethod)
	{
		CLVariable counter = currentMethod.getLocalVar("counter");
		Preconditions.checkNotNull(counter, "");
		currentMethod.addReturn(counter);
	}

	/*********************************
	 * NON-SYNCHRONIZED UPDATE
	 ********************************/
	@Override
	protected void updateMethodPerformSelection(Method currentMethod) throws KernelException
	{
		//INPUT: selectionSum - float [0, numberOfAllCommands];
		CLVariable sum = currentMethod.getArg("selectionSum");
		CLVariable number = currentMethod.getArg("numberOfCommands");
		CLVariable selection = currentMethod.getLocalVar("selection");
		CLVariable guardsTab = currentMethod.getArg("guardsTab");
		guardsTab = guardsTab.varType.accessElement(guardsTab, selection.getName());
		//currentMethod.addExpression(new Expression(
		//	"if(get_global_id(0)<10)printf(\"before %f %d %d %f\\n\",selectionSum,selection,numberOfCommands,((float)selection) / numberOfCommands);"));
		// selection = floor(selectionSum)
		//currentMethod.addExpression(ExpressionGenerator.createAssignment(selection, ExpressionGenerator.functionCall("floor", sum.getName())));

		// selectionSum = numberOfCommands * ( selectionSum - selection/numberOfCommands);
		Expression divideSelection = createBasicExpression(selection.cast("float"), Operator.DIV, number.getSource());
		Expression subSum = createBasicExpression(sum.getSource(), Operator.SUB, divideSelection);
		ExpressionGenerator.addParentheses(subSum);
		Expression asSum = createBasicExpression(number.getSource(), Operator.MUL, subSum);
		currentMethod.addExpression(ExpressionGenerator.createAssignment(sum, asSum));
	}

	@Override
	protected void updateMethodAdditionalArgs(Method currentMethod) throws KernelException
	{
		//uint numberOfCommands
		CLVariable numberOfCommands = new CLVariable(new StdVariableType(0, commands.length - 1), "numberOfCommands");
		currentMethod.addArg(numberOfCommands);
	}

	@Override
	protected void updateMethodLocalVars(Method currentMethod) throws KernelException
	{
		//selection
		CLVariable sum = currentMethod.getArg("selectionSum");
		CLVariable number = currentMethod.getArg("numberOfCommands");
		CLVariable selection = currentMethod.getLocalVar("selection");
		String selectionExpression = String.format("floor(%s)", createBasicExpression(sum.getSource(), Operator.MUL, number.getSource()).toString());
		selection.setInitValue(new Expression(selectionExpression));
	}

	/*********************************
	 * PROPERTY METHODS
	 ********************************/
	@Override
	protected void propertiesMethodTimeArg(Method currentMethod) throws KernelException
	{
		//time necessary only in case of bounded until
		if (timingProperty) {
			CLVariable time = new CLVariable(new StdVariableType(0, config.maxPathLength), "time");
			currentMethod.addArg(time);
		}
	}

	@Override
	protected void propertiesMethodAddBoundedUntil(Method currentMethod, ComplexKernelComponent parent, SamplerBoolean property, CLVariable propertyVar)
	{
		CLVariable time = currentMethod.getArg("time");
		SamplerBoundedUntilDisc prop = (SamplerBoundedUntilDisc) property;
		/**
		 * if(time > upper_bound)
		 */
		IfElse ifElse = new IfElse(createBasicExpression(time.getSource(), Operator.GE, fromString(prop.getUpperBound())));
		/**
		 * if(right_side == true) -> true
		 * else -> false
		 */
		IfElse rhsCheck = null;
		//TODO: always !prop?
		if (prop.getRightSide().toString().charAt(0) == '!') {
			rhsCheck = createPropertyCondition(propertyVar, true, prop.getRightSide().toString().substring(1), true);
		} else {
			rhsCheck = createPropertyCondition(propertyVar, false, prop.getRightSide().toString(), true);
		}
		createPropertyCondition(rhsCheck, propertyVar, false, null, false);
		ifElse.addExpression(rhsCheck);
		/**
		 * Else -> check RHS and LHS
		 */
		ifElse.addElse();
		/**
		 * if(right_side == true) -> true
		 * else if(left_side == false) -> false
		 */
		IfElse betweenBounds = null;
		if (prop.getRightSide().toString().charAt(0) == '!') {
			betweenBounds = createPropertyCondition(propertyVar, true, prop.getRightSide().toString().substring(1), true);
		} else {
			betweenBounds = createPropertyCondition(propertyVar, false, prop.getRightSide().toString(), true);
		}
		if (!(prop.getLeftSide() instanceof ExpressionLiteral)) {
			createPropertyCondition(betweenBounds, propertyVar, true, prop.getLeftSide().toString(), false);
		}
		ifElse.addExpression(1, betweenBounds);
		parent.addExpression(ifElse);
	}

	/*********************************
	 * SYNCHRONIZED GUARDS CHECK
	 ********************************/
	@Override
	protected Method guardsSynCreateMethod(String label, int maxCommandsNumber)
	{
		Method currentMethod = new Method(label, new StdVariableType(0, maxCommandsNumber));
		return currentMethod;
	}

	@Override
	protected CLVariable guardsSynLabelVar(int maxCommandsNumber)
	{
		return new CLVariable(new StdVariableType(0, maxCommandsNumber), "labelSize");
	}

	@Override
	protected CLVariable guardsSynCurrentVar(int maxCommandsNumber)
	{
		return new CLVariable(new StdVariableType(0, maxCommandsNumber), "currentSize");
	}

	@Override
	protected void guardsSynAddGuard(ComplexKernelComponent parent, CLVariable guardArray, Command cmd, CLVariable size)
	{
		Expression guard = new Expression(convertPrismGuard(svVars, cmd.getGuard().toString()));
		parent.addExpression(createAssignment(guardArray, guard));
		parent.addExpression(createBasicExpression(size.getSource(), Operator.ADD_AUGM,
		//converted guard
				guardArray.getSource()));
	}

	/*********************************
	 * SYNCHRONIZED UPDATE
	 ********************************/
	@Override
	protected void updateSynAdditionalVars(Method parent, SynchronizedCommand cmd)
	{

	}

	protected void updateSynBeforeUpdateLabel(Method parent, SynchronizedCommand cmd, int moduleNumber, CLVariable guardsTab, CLVariable guard,
			CLVariable moduleSize, CLVariable totalSize, CLVariable probability)
	{
		/**
		 * compute current guard in update
		 */
		Expression guardUpdate = functionCall("floor",
		//probability * module_size
				createBasicExpression(probability.getSource(), Operator.MUL, moduleSize.getSource()));
		parent.addExpression(createAssignment(guard, guardUpdate));
		/**
		* recompute probability to an [0,1) in selected guard
		*/
		Expression probUpdate = createBasicExpression(
		//probability * module_size
				createBasicExpression(probability.getSource(), Operator.MUL, moduleSize.getSource()), Operator.SUB,
				//guard
				guard.getSource());
		parent.addExpression(createAssignment(probability, probUpdate));
		//current.addExpression(new Expression("if(get_global_id(0)<5)printf(\"" + cmd.synchLabel + " %f %d %d\\n\",prop,guard,(*sv).__STATE_VECTOR_q);"));

	}

	protected void updateSynAfterUpdateLabel(ComplexKernelComponent parent, CLVariable guard, CLVariable moduleSize, CLVariable totalSize,
			CLVariable probability)
	{
		/**
		 * totalSize /= 1 is useless
		 */
		parent.addExpression(createBasicExpression(totalSize.getSource(), Operator.DIV_AUGM, moduleSize.getSource()));
	}

	protected CLVariable updateSynLabelMethodGuardCounter(SynchronizedCommand cmd)
	{
		CLVariable guardCounter = new CLVariable(new StdVariableType(-1, cmd.getCmdsNum()), "guardCounter");
		guardCounter.setInitValue(StdVariableType.initialize(-1));
		return guardCounter;
	}

	protected CLVariable updateSynLabelMethodGuardSelection(SynchronizedCommand cmd, CLVariable guard)
	{
		CLVariable guardCounter = new CLVariable(new StdVariableType(0, cmd.getCmdsNum()), "guardSelection");
		guardCounter.setInitValue(StdVariableType.initialize(0));
		return guardCounter;
	}

	protected void updateSynLabelMethodSelectGuard(Method currentMethod, ComplexKernelComponent parent, CLVariable guardSelection, CLVariable guardCounter,
			int moduleOffset)
	{
		CLVariable guardsTab = currentMethod.getArg("guards");
		CLVariable guard = currentMethod.getArg("guard");
		/**
		 * guardSelection = -1;
		 * guardCounter = 0;
		 * for(;;guardCounter++) {
		 * 	guardSelection += guards[moduleOffset+guardCounter];
		 * 	if(guardsSelection == guard)
		 * 		break;
		 * }
		 * switch(guardCounter)...
		 */
		ForLoop guardSelectionLoop = new ForLoop(guardSelection, false);
		CLVariable guardsAccess = guardsTab.accessElement(
		// moduleOffset(constant!) + guardCounter
				createBasicExpression(fromString(moduleOffset), Operator.ADD, guardSelection.getSource()));
		guardSelectionLoop.addExpression(createBasicExpression(guardCounter.getSource(),
		//guardSelection += guards[moduleOffset + guardCounter];
				Operator.ADD_AUGM, guardsAccess.getSource()));
		IfElse ifElseGuardLoop = new IfElse(createBasicExpression(guardCounter.getSource(),
		//guardSelection == guard
				Operator.EQ, guard.getSource()));
		ifElseGuardLoop.addExpression("break\n");
		guardSelectionLoop.addExpression(ifElseGuardLoop);
		parent.addExpression(guardSelectionLoop);
		//		parent.addExpression(new Expression("if(get_global_id(0)<5)printf(\"" + synCmd.synchLabel + " %d %d \\n\"," + guardCounter.varName + ","
		//				+ guardSelection.varName + ");"));

	}

	@Override
	protected Expression updateSynProbabilityRecompute(CLVariable probability, Rate before, Rate current)
	{
		Expression compute = null;
		if (before != null) {
			compute = createBasicExpression(probability.getSource(), Operator.SUB,
			//probability - sum of rates before
					fromString(convertPrismRate(svVars, before)));
		} else {
			compute = probability.getSource();
		}
		addParentheses(compute);
		return createAssignment(probability, createBasicExpression(compute, Operator.DIV,
		//divide by current interval
				fromString(convertPrismRate(svVars, current))));
	}
}