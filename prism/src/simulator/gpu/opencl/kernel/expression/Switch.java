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
package simulator.gpu.opencl.kernel.expression;

import java.util.ArrayList;
import java.util.List;

import prism.Preconditions;
import simulator.gpu.opencl.kernel.memory.CLVariable;

public class Switch extends ComplexKernelComponent
{
	private static class Case implements KernelComponent
	{
		public final Expression value;
		public List<KernelComponent> commands = new ArrayList<>();

		public Case()
		{
			value = null;
		}

		public Case(Expression value)
		{
			this.value = value;
		}

		@Override
		public boolean hasIncludes()
		{
			return false;
		}

		@Override
		public boolean hasDeclaration()
		{
			return false;
		}

		@Override
		public List<Include> getIncludes()
		{
			return null;
		}

		@Override
		public KernelComponent getDeclaration()
		{
			return null;
		}

		@Override
		public void accept(VisitorInterface v)
		{
			value.accept(v);
			for (KernelComponent command : commands) {
				command.accept(v);
			}
		}

		@Override
		public String getSource()
		{
			StringBuilder builder = new StringBuilder();
			if (value != null) {
				builder.append("case ").append(value.getSource()).append(":\n");
			} else {
				builder.append("default:\n ");
			}
			for (KernelComponent command : commands) {
				builder.append(command.getSource()).append("\n");
			}
			builder.append("break;");
			return builder.toString();
		}
	}

	private boolean hasDefault = false;
	private Expression switchCondition = null;
	private int conditionNumber = 0;

	public Switch(Expression switchCondition)
	{
		this.switchCondition = switchCondition;
	}

	public Switch(CLVariable var)
	{
		this.switchCondition = new Expression(var.varName);
	}

	public void addCase(Expression condition)
	{
		if (hasDefault) {
			body.add(body.size() - 1, new Case(condition));
		} else {
			body.add(new Case(condition));
		}
	}

	public void addDefault()
	{
		hasDefault = true;
	}

	public void addExpression(int conditionNumber, KernelComponent command)
	{
		Preconditions.checkIndex(conditionNumber, body.size(), "Non-valid index of condition in Switch!");
		correctExpression(command);
		((Case) body.get(conditionNumber)).commands.add(command);
	}

	public void addExpression(KernelComponent expr)
	{
		Preconditions.checkNotNull(expr, "Trying to add null reference to expression!");
		correctExpression(expr);
		addExpression(conditionNumber, expr);
		if (expr.hasIncludes()) {
			necessaryIncludes.addAll(expr.getIncludes());
		}
	}

	public void setConditionNumber(int conditionNumber)
	{
		Preconditions.checkIndex(conditionNumber, body.size(), "Non-valid index of condition in IfElse!");
		this.conditionNumber = conditionNumber;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.expression.ComplexKernelComponent#createHeader()
	 */
	@Override
	protected String createHeader()
	{
		return null;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.expression.ComplexKernelComponent#hasDeclaration()
	 */
	@Override
	public boolean hasDeclaration()
	{
		return false;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.expression.ComplexKernelComponent#getDeclaration()
	 */
	@Override
	public KernelComponent getDeclaration()
	{
		return null;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.expression.ComplexKernelComponent#accept(simulator.gpu.opencl.kernel.expression.VisitorInterface)
	 */
	@Override
	public void accept(VisitorInterface v)
	{
		for (KernelComponent component : body) {
			component.accept(v);
		}
	}

	@Override
	public String getSource()
	{
		StringBuilder source = new StringBuilder("switch(");
		source.append(switchCondition).append("){\n");
		for (KernelComponent e : body) {
			source.append(e.getSource()).append("\n");
		}
		source.append("}\n");
		return source.toString();
	}
}
