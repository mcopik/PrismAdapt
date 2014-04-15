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
package simulator.gpu.automaton;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import parser.VarList;
import parser.ast.Module;
import parser.ast.ModulesFile;
import prism.Preconditions;
import prism.PrismException;
import simulator.gpu.automaton.command.CommandBuilder;
import simulator.gpu.automaton.command.CommandInterface;

public abstract class AbstractAutomaton
{
	public enum AutomatonType {
		DTMC, CTMC
	}

	protected ModulesFile modulesFile = null;
	protected VarList varList = null;
	protected List<CommandInterface> commands = new ArrayList<>();
	protected StateVector variables = new StateVector();
	protected int numOfCommands = 0;
	protected int numOfSyncCommands = 0;
	protected int numOfAdaptCommands = 0;

	public class StateVector
	{
		protected Map<String, PrismVariable> variables = new HashMap<>();

		public void addVariable(PrismVariable var)
		{
			variables.put(var.name, var);
		}

		public PrismVariable getVar(String name)
		{
			return variables.get(name);
		}

		public PrismVariable[] getVars()
		{
			Collection<PrismVariable> vars = variables.values();
			return vars.toArray(new PrismVariable[vars.size()]);
		}

		public Set<String> getNames()
		{
			return variables.keySet();
		}

		public int size()
		{
			return variables.size();
		}

		public String toString()
		{
			StringBuilder builder = new StringBuilder();
			for (Map.Entry<String, PrismVariable> var : variables.entrySet()) {
				builder.append(var.getValue().toString() + "\n");
			}
			return builder.toString();
		}
	}

	public AbstractAutomaton(ModulesFile modulesFile) throws PrismException
	{
		varList = modulesFile.createVarList();
		modulesFile = (ModulesFile) modulesFile.deepCopy().replaceConstants(modulesFile.getConstantValues());
		//TODO: error in nand.pm
		//modulesFile.simplify();
		this.modulesFile = modulesFile;
		this.modulesFile.tidyUp();
		extractVariables();
		extractUpdates();
	}

	public int synchCmdsNumber()
	{
		return numOfSyncCommands;
	}

	public int adaptCmdsNumber()
	{
		return numOfAdaptCommands;
	}

	public int commandsNumber()
	{
		return commands.size();
	}

	public CommandInterface getCommand(int number)
	{
		Preconditions.checkIndex(number, commands.size(), String.format("Command number %d pis bigger than commands size", number));
		return commands.get(number);
	}

	protected void extractVariables()
	{
		int varLen = varList.getNumVars(), i = 0;
		for (i = 0; i < varLen; ++i) {
			PrismVariable variable = new PrismVariable(varList.getName(i), varList.getLow(i), varList.getHigh(i), varList.getStart(i),
					varList.getRangeLogTwo(i));
			variables.addVariable(variable);
		}
	}

	protected void extractUpdates()
	{
		CommandBuilder builder = new CommandBuilder(getType(), variables);
		for (String moduleName : modulesFile.getModuleNames()) {
			Module currentModule = modulesFile.getModule(modulesFile.getModuleIndex(moduleName));
			builder.setCurrentModule(currentModule.getName());
			for (int i = 0; i < currentModule.getNumCommands(); ++i) {
				builder.addCommand(currentModule.getCommand(i).getSynch(), currentModule.getCommand(i).getGuard(), currentModule.getCommand(i).getUpdates());
			}
		}
		numOfSyncCommands = builder.synchCmdsNumber();
		numOfAdaptCommands = builder.adaptCmdsNumber();
		commands = builder.getResults();
	}

	/**
	 * 
	 * @return
	 */
	public abstract AutomatonType getType();

	public StateVector getStateVector()
	{
		return variables;
	}

	@Override
	public String toString()
	{
		StringBuilder builder = new StringBuilder("DTMC\n");
		builder.append("Variables: \n");
		builder.append(variables);
		builder.append("Commands: \n");
		for (CommandInterface cmd : commands) {
			builder.append(cmd).append("\n");
		}
		return builder.toString();
	}
}