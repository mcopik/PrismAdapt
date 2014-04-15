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
package simulator.gpu.automaton.command;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import prism.PrismException;
import simulator.gpu.automaton.Guard;
import simulator.gpu.automaton.update.Rate;
import simulator.gpu.automaton.update.Update;

public class SynchronizedCommand implements CommandInterface
{
	public class ModuleGroup
	{
		public final String moduleName;
		Rate rate = new Rate();
		private List<Command> cmds = new ArrayList<>();

		public ModuleGroup(String name)
		{
			moduleName = name;
		}

		public void addCommand(Command cmd)
		{
			cmds.add(cmd);
			rate.addRate(cmd.getRateSum());
		}

		public int getCommandsNum()
		{
			return cmds.size();
		}
	}

	public final String synchLabel;
	private Map<String, ModuleGroup> synchronizedCommands = new HashMap<>();
	private List<String> moduleNames = new ArrayList<>();

	public SynchronizedCommand(String label)
	{
		synchLabel = label;
	}

	public int getModulesNum()
	{
		return synchronizedCommands.size();
	}

	public void addCommand(String moduleName, Command cmd)
	{
		getModule(moduleName).addCommand(cmd);
		if (moduleNames.size() != synchronizedCommands.size()) {
			moduleNames.add(moduleName);
		}
	}

	private ModuleGroup getModule(String moduleName)
	{
		ModuleGroup group = null;
		if (synchronizedCommands.containsKey(moduleName)) {
			group = synchronizedCommands.get(moduleName);
		} else {
			group = new ModuleGroup(moduleName);
			synchronizedCommands.put(moduleName, group);
		}
		return group;
	}

	public Command getCommand(int module, int command)
	{
		return synchronizedCommands.get(moduleNames.get(module)).cmds.get(command);
	}

	public int getCommandNumber(int module)
	{
		return synchronizedCommands.get(moduleNames.get(module)).getCommandsNum();
	}

	@Override
	public Guard getGuard()
	{
		throw new IllegalAccessError("Method getGuard is not " + "defined for type SynchronizedCommand");
	}

	@Override
	public Update getUpdate()
	{
		throw new IllegalAccessError("Method getUpdate is not " + "defined for type SynchronizedCommand");
	}

	@Override
	public boolean isSynchronized()
	{
		return true;
	}

	public Rate getRateSumUpdate(int module, int update) throws PrismException
	{
		return synchronizedCommands.get(module).cmds.get(update).getRateSum();
	}

	public Rate getRateSumModule(int i) throws PrismException
	{
		return synchronizedCommands.get(i).rate;
	}

	public int getUpdateNumberModule(int i) throws PrismException
	{
		return synchronizedCommands.get(i).getCommandsNum();
	}

	public String toString()
	{
		StringBuilder builder = new StringBuilder();
		builder.append("SYNCHRONIZED COMMAND: ").append(synchLabel).append("\n");
		for (Map.Entry<String, ModuleGroup> group : synchronizedCommands.entrySet()) {
			builder.append(group.getKey()).append(" - sum: ").append(group.getValue().rate).append("\n");
			for (Command cmd : group.getValue().cmds) {
				builder.append("Update sum: ").append(cmd.getRateSum()).append("; ").append(cmd).append("\n");
			}
		}
		return builder.toString();
	}

	@Override
	public Rate getRateSum()
	{
		Rate sum = new Rate(0);
		for (Map.Entry<String, ModuleGroup> group : synchronizedCommands.entrySet()) {
			sum.addRate(group.getValue().rate);
		}
		return sum;
	}

	public int getMaxCommandsNum()
	{
		int cmdNumber = 0, sum = 1;
		for (int i = 0; i < getModulesNum(); ++i) {
			cmdNumber = getCommandNumber(i);
			sum *= cmdNumber;
		}
		return sum;
	}

	public int getCmdsNum()
	{
		int cmdNumber = 0;
		for (int i = 0; i < getModulesNum(); ++i) {
			cmdNumber += getCommandNumber(i);
		}
		return cmdNumber;
	}
}