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
package simulator.gpu;

import java.util.ArrayList;
import java.util.List;

import parser.State;
import parser.ast.Expression;
import parser.ast.ExpressionFilter;
import parser.ast.ExpressionProb;
import parser.ast.ExpressionReward;
import parser.ast.ExpressionTemporal;
import parser.ast.LabelList;
import parser.ast.ModulesFile;
import parser.ast.PropertiesFile;
import prism.ModelType;
import prism.Prism;
import prism.PrismException;
import prism.PrismLog;
import prism.ResultsCollection;
import prism.UndefinedConstants;
import simulator.ModelCheckInterface;
import simulator.gpu.automaton.AbstractAutomaton;
import simulator.gpu.automaton.CTMC;
import simulator.gpu.automaton.DTMC;
import simulator.method.SimulationMethod;
import simulator.sampler.Sampler;

public class GPUSimulatorEngine implements ModelCheckInterface
{
	/**
	 * Main Prism object.
	 */
	protected Prism prism;
	/**
	 * Prism main log.
	 */
	protected PrismLog mainLog;
	/**
	 * Represents Prism's object of current automaton.
	 */
	private ModulesFile modulesFile;
	/**
	 * Current automaton structure for GPU.
	 */
	private AbstractAutomaton automaton;
	/**
	 * Samplers for properties.
	 * null if the property cannot be handled
	 */
	private Sampler[] samplers = null;
	/**
	 * Results of simulation.
	 * Exception if the property cannot be handled.
	 */
	private Object[] results = null;
	/**
	 * Simulation framework, used for simulation.
	 */
	private RuntimeFrameworkInterface simFramework;

	/**
	 * Main constructor
	 * @param framework
	 * @param prism
	 */
	public GPUSimulatorEngine(RuntimeFrameworkInterface framework, Prism prism)
	{
		this.prism = prism;
		mainLog = prism.getLog();
		simFramework = framework;
	}

	/**
	 * Create automaton object using PRISM object
	 * @param modulesFile
	 * @throws PrismException
	 */
	private void loadModel(ModulesFile modulesFile) throws PrismException
	{
		checkModelForSimulation(modulesFile);
		if (modulesFile.getModelType() == ModelType.DTMC || modulesFile.getModelType() == ModelType.MDP) {
			automaton = new DTMC(modulesFile);
		} else if (modulesFile.getModelType() == ModelType.CTMC) {
			automaton = new CTMC(modulesFile);
		}
		this.modulesFile = modulesFile;
	}

	/**
	 * Create samplers for properties using selected simulation method.
	 * @param properties
	 * @param pf
	 * @param simMethod
	 * @throws PrismException
	 */
	private void createSamplers(List<Expression> properties, PropertiesFile pf, SimulationMethod simMethod) throws PrismException
	{
		simMethod.computeMissingParameterBeforeSim();
		samplers = new Sampler[properties.size()];
		results = new Object[properties.size()];
		/**
		 * Code taken from SimulatorEngine.java
		 */
		for (int i = 0; i < properties.size(); ++i) {
			// Take a copy
			Expression propNew = properties.get(i).deepCopy();
			// Combine label lists from model/property file, then expand property refs/labels in property 
			LabelList combinedLabelList = (pf == null) ? modulesFile.getLabelList() : pf.getCombinedLabelList();
			// formulas must be expanded before replacing constants!!!
			propNew = (Expression) propNew.expandFormulas(modulesFile.getFormulaList());
			propNew = (Expression) propNew.expandPropRefsAndLabels(pf, combinedLabelList);
			// Then get rid of any constants and simplify
			propNew = (Expression) propNew.replaceConstants(modulesFile.getConstantValues());
			if (pf != null) {
				propNew = (Expression) propNew.replaceConstants(pf.getConstantValues());
			}
			//TODO: removing parentheses breaks things
			//propNew = (Expression) propNew.simplify();
			Sampler sampler = Sampler.createSampler(propNew, modulesFile);
			SimulationMethod simMethodNew = simMethod.clone();
			sampler.setSimulationMethod(simMethodNew);
			// Pass property details to SimuationMethod
			// (note that we use the copy stored in properties, which has been processed)
			try {
				simMethodNew.setExpression(properties.get(i));
			} catch (PrismException e) {
				results[i] = e;
				samplers[i] = null;
				continue;
			}
			results[i] = null;
			samplers[i] = sampler;
		}
	}

	/**
	 * Change the framework used for simulation.
	 * @param framework
	 */
	public void setSimulatorFramework(RuntimeFrameworkInterface framework)
	{
		simFramework = framework;
	}

	@Override
	public void checkModelForSimulation(ModulesFile modulesFile) throws PrismException
	{
		if (modulesFile.getModelType() != ModelType.DTMC && modulesFile.getModelType() != ModelType.CTMC && modulesFile.getModelType() != ModelType.MDP) {
			throw new PrismException("Currently only DTMC/CTMC is supported!");
		}
	}

	@Override
	public boolean isPropertyOKForSimulation(Expression expr)
	{
		return isPropertyOKForSimulationString(expr) == null;
	}

	@Override
	public void checkPropertyForSimulation(Expression expr) throws PrismException
	{
		String explanation = isPropertyOKForSimulationString(expr);
		if (explanation != null) {
			throw new PrismException(explanation);
		}
	}

	/**
	 * Function was taken from SimulationEngine.java
	 * @param expr
	 * @return string with explanation if the property cannot be simulated; otherwise null
	 */
	private String isPropertyOKForSimulationString(Expression expr)
	{
		// Simulator can only be applied to P or R properties (without filters)
		if (!(expr instanceof ExpressionProb || expr instanceof ExpressionReward)) {
			if (expr instanceof ExpressionFilter) {
				if (((ExpressionFilter) expr).getOperand() instanceof ExpressionProb || ((ExpressionFilter) expr).getOperand() instanceof ExpressionReward)
					return "Simulator cannot handle P or R properties with filters";
			}
			return "Simulator can only handle P or R properties";
		}
		// Check that there are no nested probabilistic operators
		try {
			if (expr.computeProbNesting() > 1) {
				return "Simulator cannot handle nested P, R or S operators";
			}
		} catch (PrismException e) {
			return "Simulator cannot handle this property: " + e.getMessage();
		}
		// Simulator cannot handle cumulative reward properties without a time bound
		if (expr instanceof ExpressionReward) {
			Expression exprTemp = ((ExpressionReward) expr).getExpression();
			if (exprTemp instanceof ExpressionTemporal) {
				if (((ExpressionTemporal) exprTemp).getOperator() == ExpressionTemporal.R_C) {
					if (((ExpressionTemporal) exprTemp).getUpperBound() == null) {
						return "Simulator cannot handle cumulative reward properties without time bounds";
					}
				}
			}
		}
		// No errors
		return null;
	}

	@Override
	public Object modelCheckSingleProperty(ModulesFile modulesFile, PropertiesFile propertiesFile, Expression expr, State initialState, int maxPathLength,
			SimulationMethod simMethod) throws PrismException
	{
		List<Expression> exprs = new ArrayList<>();
		exprs.add(expr);
		return modelCheckMultipleProperties(modulesFile, propertiesFile, exprs, initialState, maxPathLength, simMethod)[0];

	}

	@Override
	public Object[] modelCheckMultipleProperties(ModulesFile modulesFile, PropertiesFile propertiesFile, List<Expression> exprs, State initialState,
			int maxPathLength, SimulationMethod simMethod) throws PrismException
	{
		loadModel(modulesFile);
		createSamplers(exprs, propertiesFile, simMethod);
		List<Sampler> validSamplers = new ArrayList<>();
		for (Sampler sampler : samplers) {
			if (sampler != null) {
				validSamplers.add(sampler);
			}
		}
		if (initialState != null) {
			simFramework.setInitialState(initialState);
		}
		simFramework.setMaxPathLength(maxPathLength);
		simFramework.setMainLog(mainLog);
		simFramework.setPrismSettings(prism.getSettings());
		simFramework.setMF(modulesFile);
		simFramework.setPF(propertiesFile);
		simFramework.setExprs(exprs.get(0));
		int samplesProcessed = simFramework.simulateProperty(automaton, validSamplers);
		for (int i = 0; i < samplers.length; i++) {
			Sampler sampler = samplers[i];
			if (sampler != null) {
				SimulationMethod sm = sampler.getSimulationMethod();
				//TODO: temporal fix to avoid wrong width computation
				sm.shouldStopNow(samplesProcessed, sampler);
				sm.computeMissingParameterAfterSim();
				try {
					results[i] = sm.getResult(sampler);
				} catch (PrismException e) {
					results[i] = e;
				}
			}
		}

		/**
		 * Code taken from SimulatorEngine.java
		 */

		// Display results to log
		if (results.length == 1) {
			mainLog.print("\nSimulation method parameters: ");
			mainLog.println((samplers[0] == null) ? "no simulation" : samplers[0].getSimulationMethod().getParametersString());
			mainLog.print("\nSimulation result details: ");
			mainLog.println((samplers[0] == null) ? "no simulation" : samplers[0].getSimulationMethodResultExplanation());
			if (!(results[0] instanceof PrismException))
				mainLog.println("\nResult: " + results[0]);
		} else {
			mainLog.println("\nSimulation method parameters:");
			for (int i = 0; i < results.length; i++) {
				mainLog.print(exprs.get(i) + " : ");
				mainLog.println((samplers[i] == null) ? "no simulation" : samplers[i].getSimulationMethod().getParametersString());
			}
			mainLog.println("\nSimulation result details:");
			for (int i = 0; i < results.length; i++) {
				mainLog.print(exprs.get(i) + " : ");
				mainLog.println((samplers[i] == null) ? "no simulation" : samplers[i].getSimulationMethodResultExplanation());
			}
			mainLog.println("\nResults:");
			for (int i = 0; i < results.length; i++)
				mainLog.println(exprs.get(i) + " : " + results[i]);
		}
		return results;
	}

	@Override
	public void modelCheckExperiment(ModulesFile modulesFile, PropertiesFile propertiesFile, UndefinedConstants undefinedConstants,
			ResultsCollection resultsCollection, Expression expr, State initialState, int maxPathLength, SimulationMethod simMethod) throws PrismException,
			InterruptedException
	{
		//TODO
	}
}